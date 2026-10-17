// PufferNet model API + architecture
// Writing custom nets in 4.0+ requires a fair bit of code because you are
// responsible for defining your own activation and gradient buffers.
// You usually only ever need a custom Encoder.
#include <cooperative_groups.h>

typedef void (*init_weights_fn)(void* weights, ulong* seed, cudaStream_t stream);
typedef void (*reg_params_fn)(void* weights, Allocator* alloc);
typedef void (*reg_train_fn)(void* weights, void* buf, Allocator* acts, Allocator* grads, int B_TT);
typedef void (*reg_rollout_fn)(void* weights, void* buf, Allocator* alloc, int B);
typedef void* (*create_weights_fn)(void* self);
typedef Prec (*forward_fn)(void* weights, void* activations, Prec input, cudaStream_t stream);
typedef void (*encoder_backward_fn)(void* weights, void* activations,
    Prec grad, cudaStream_t stream);
typedef Prec (*decoder_backward_fn)(void* weights, void* activations,
    Float grad_logits, Float grad_logstd, Float grad_value, cudaStream_t stream);
typedef Prec (*network_forward_fn)(void* weights, Prec x,
    Prec state, void* activations, cudaStream_t stream);
typedef Prec (*network_forward_train_fn)(void* weights, Prec x,
    Prec state, Prec terminals, void* activations, int agent_off,
    cudaStream_t stream);
typedef Prec (*network_backward_fn)(void* weights,
    Prec grad, void* activations, cudaStream_t stream);

struct Encoder {
    forward_fn forward;
    encoder_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    int in_dim, out_dim;
    size_t activation_size;  // sizeof(EncoderActivations) or custom override
};

struct EncoderWeights {
    Prec weight;
    int in_dim, out_dim;
};

struct EncoderActivations {
    Prec out, saved_input, wgrad_scratch;
};

struct Decoder {
    forward_fn forward;
    decoder_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    int hidden_dim, output_dim;
    bool continuous;
    size_t activation_size;
};

struct DecoderWeights {
    Prec weight, logstd;
    int hidden_dim, output_dim;
    bool continuous;
};

struct DecoderActivations {
    Prec out, grad_out, saved_input, grad_input, wgrad_scratch, logstd_scratch;
};

struct Network {
    network_forward_fn forward;
    network_forward_train_fn forward_train;
    network_backward_fn backward;
    init_weights_fn init_weights;
    reg_params_fn reg_params;
    reg_train_fn reg_train;
    reg_rollout_fn reg_rollout;
    create_weights_fn create_weights;
    int hidden, num_layers, horizon;
};

thread_local cublasHandle_t g_cublas_handle = NULL;
thread_local void* g_cublas_workspace = NULL;
// Side-stream dW (mm_tn) overlaps dX (mm_nn) on main during linear bwd.
thread_local cublasHandle_t g_cublas_dw_handle = NULL;
thread_local cudaEvent_t g_main_ready = NULL;
thread_local void* g_cublas_dw_workspace = NULL;
thread_local cudaStream_t g_dw_stream = NULL;
thread_local cudaEvent_t g_dw_done = NULL;

#define CUBLAS_WS_BYTES (32 * 1024 * 1024)

static void cublas_init_one(cublasHandle_t* handle, void** workspace) {
    cublasCreate(handle);
    cudaMalloc(workspace, CUBLAS_WS_BYTES);
    cublasSetMathMode(*handle, CUBLAS_DEFAULT_MATH);
}

void cublas_init_handle() {
    cublas_init_one(&g_cublas_handle, &g_cublas_workspace);
    cublas_init_one(&g_cublas_dw_handle, &g_cublas_dw_workspace);
    // Backward-only side stream: same top priority as the async learner stream.
    int prio_low, prio_high;
    cudaDeviceGetStreamPriorityRange(&prio_low, &prio_high);
    cudaStreamCreateWithPriority(&g_dw_stream, cudaStreamNonBlocking, prio_high);
    cudaEventCreateWithFlags(&g_dw_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&g_main_ready, cudaEventDisableTiming);
}

// Dense row-major GEMM: C = alpha * op_a(A) @ op_b(B) + beta * C
static void cublasGemmExDense(cublasHandle_t handle,
        cublasOperation_t op_a, cublasOperation_t op_b,
        int M, int N, int K, void* A, void* B, void* C,
        cudaStream_t stream, float alpha = 1.0f, float beta = 0.0f) {
    int lda = (op_a == CUBLAS_OP_N) ? K : M;
    int ldb = (op_b == CUBLAS_OP_N) ? N : K;
    // cublasSetStream resets the handle to cuBLAS's default workspace pool;
    // on that pool async + graph runs were not bitwise reproducible. Re-pin the
    // handle's own workspace after every stream switch.
    cublasSetStream(handle, stream);
    cublasSetWorkspace(handle, handle == g_cublas_handle
        ? g_cublas_workspace : g_cublas_dw_workspace, CUBLAS_WS_BYTES);
    cublasGemmEx(handle, op_b, op_a, N, M, K, &alpha,
        B, CUBLAS_PRECISION, ldb, A, CUBLAS_PRECISION, lda, &beta,
        C, CUBLAS_PRECISION, N, CUBLAS_COMPUTE, CUBLAS_GEMM_DEFAULT);
}

// out(...,N) = alpha * a(...,K) @ b(N,K)^T + beta * out: leading dims folded into M
void puf_mm(Prec* a, Prec* b, Prec* out, cudaStream_t stream,
        float alpha = 1.0f, float beta = 0.0f) {
    int M = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int K = a->shape[ndim(a->shape)-1];
    int N = b->shape[ndim(b->shape)-2];
    cublasGemmExDense(g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, M, N, K,
        a->data, b->data, out->data, stream, alpha, beta);
}

// out(M,N) = alpha * a(...,M)^T @ b(...,N) + beta * out: leading dims folded into K
void puf_mm_tn(Prec* a, Prec* b, Prec* out, cudaStream_t stream,
        float alpha = 1.0f, float beta = 0.0f,
        cublasHandle_t handle = g_cublas_handle) {
    int M = a->shape[ndim(a->shape)-1];
    int K = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int N = b->shape[ndim(b->shape)-1];
    cublasGemmExDense(handle, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K,
        a->data, b->data, out->data, stream, alpha, beta);
}

// out(...,N) = alpha * a(...,K) @ b(K,N) + beta * out: leading dims folded into M
void puf_mm_nn(Prec* a, Prec* b, Prec* out, cudaStream_t stream,
        float alpha = 1.0f, float beta = 0.0f) {
    int M = batch_size(a->shape) * a->shape[ndim(a->shape)-2];
    int K = a->shape[ndim(a->shape)-1];
    int N = b->shape[ndim(b->shape)-1];
    cublasGemmExDense(g_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K,
        a->data, b->data, out->data, stream, alpha, beta);
}

// The side stream, ordered after the work queued so far on main. Side work must not
// gate main; puf_dw_join (before muon) joins it all.
cudaStream_t puf_fork(cudaStream_t main_stream) {
    cudaEventRecord(g_main_ready, main_stream);
    cudaStreamWaitEvent(g_dw_stream, g_main_ready, 0);
    return g_dw_stream;
}

// Queue dW (mm_tn) on the side stream. Per-layer wgrad buffers are disjoint so
// multiple dWs can queue.
void puf_mm_tn_async_after(Prec* a, Prec* b, Prec* out, cudaStream_t main_stream) {
    puf_mm_tn(a, b, out, puf_fork(main_stream), 1.0f, 0.0f, g_cublas_dw_handle);
}

void puf_dw_join(cudaStream_t consumer) {
    cudaEventRecord(g_dw_done, g_dw_stream);
    cudaStreamWaitEvent(consumer, g_dw_done, 0);
}

// Weight init (needs cast, grid_size, numel/ndim from pufferl substrate).
__global__ void uniform_scale_kernel(float* data, float bound, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f * bound - bound;
    }
}

// Uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))
void puf_kaiming_init(Prec* dst,
        float gain, ulong seed, cudaStream_t stream) {
    assert(ndim(dst->shape) == 2);
    long rows = dst->shape[0], cols = dst->shape[1];
    assert(rows > 0 && cols > 0);
    long n = rows * cols;
    float bound = gain / sqrtf((float)cols);
    float* buf;
    cudaMalloc(&buf, n * sizeof(float));
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, seed);
    curandGenerateUniform(gen, buf, n);
    curandDestroyGenerator(gen);
    uniform_scale_kernel<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(buf, bound, n);
    cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(dst->data, buf, n);
    cudaFree(buf);
}

__device__ __forceinline__ float sigmoid(float x) {
    float z = expf(-fabsf(x));
    return x >= 0.0f ? 1.0f / (1.0f + z) : z / (1.0f + z);
}

__device__ __forceinline__ float lerp(float a, float b, float w) {
    float diff = b - a;
    return (fabsf(w) < 0.5f) ? a + w * diff : b - diff * (1.0f - w);
}

#if !defined(PRECISION_FLOAT)
// d (16x8 fp32) += a (16x16 bf16, row-major) * b (16x8 bf16, col-major). Lane
// 4g + t holds a: (g, 2t..2t+1), (g+8, 2t..), (g, 2t+8..), (g+8, 2t+8..);
// b: (2t..2t+1, g), (2t+8.., g); d: (g, 2t), (g, 2t+1), (g+8, 2t), (g+8, 2t+1).
// sm_80+: callers check the device before launching kernels that use it.
__device__ __forceinline__ void mma_16816(float* d, const unsigned int* a,
        unsigned int b0, unsigned int b1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}
#endif

// Rollout MinGRU step: h = (1-z)*h + z*h_tilde, highway out = s*h + (1-s)*x.
__global__ void mingru_gate(precision_t* out, precision_t* next_state,
        const precision_t* combined, const precision_t* state_in,
        const precision_t* x_in, int H, int B) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = B * H;
    if (idx >= N) {
        return;
    }

    int b = idx / H;
    int h = idx % H;

    int combined_base = b * 3 * H;
    float hidden = to_float(combined[combined_base + h]);
    float gate = to_float(combined[combined_base + H + h]);
    float proj = to_float(combined[combined_base + 2*H + h]);
    float state = to_float(state_in[idx]);
    float x = to_float(x_in[idx]);

    float z = sigmoid(gate);
    float h_tilde = (hidden >= 0.0f) ? hidden + 0.5f : sigmoid(hidden);
    float h_out = lerp(state, h_tilde, z);
    next_state[idx] = from_float(h_out);

    float s = sigmoid(proj);
    out[idx] = from_float(s * h_out + (1.0f - s) * x);
}

// Train scan buffers. Cache post-reset h inputs; recompute outputs in bwd.
struct PrefixScan {
    precision_t* combined_ptr = NULL;
    precision_t* state_ptr = NULL;
    precision_t* input_ptr = NULL;  // (B, T, H) pre-projection input (highway)
    precision_t* terminals_ptr = NULL;  // (B, T), reset before step t if terminal[t]
    int B = 0, T = 0, H = 0;
    Prec scan_h;  // (B, T, H), recurrent input to each step
    Prec out, next_state;
    Prec grad_combined, grad_state;
    Prec grad_input;
};

// Train scan: match rollout's precision_t recurrent state between steps.
__global__ void mingru_scan_forward(PrefixScan scan) {
    int T_seq = scan.T, H = scan.H, B = scan.B;
    precision_t* __restrict__ out = scan.out.data;
    precision_t* __restrict__ next_state = scan.next_state.data;
    precision_t* __restrict__ scan_h = scan.scan_h.data;
    const precision_t* __restrict__ combined = scan.combined_ptr;
    const precision_t* __restrict__ state = scan.state_ptr;
    const precision_t* __restrict__ input = scan.input_ptr;
    const precision_t* __restrict__ terminals = scan.terminals_ptr;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }

    int b = idx / H;
    int h = idx % H;

    int bH = b * H;
    int H3 = 3 * H;
    int H2 = 2 * H;
    int bHT = bH * T_seq;
    const int out_base = bHT + h;
    int cbase = 3 * bHT;

    // h_0
    float h_t = to_float(state[bH + h]);
    int h_base = b * T_seq * H + h;

    const precision_t* __restrict__ combined_h_base = &combined[cbase + h];
    const precision_t* __restrict__ combined_g_base = &combined[cbase + H + h];
    const precision_t* __restrict__ combined_p_base = &combined[cbase + H2 + h];

    int out_curr = out_base;
    int t_offset = 0;

    for (int t = 0; t < T_seq; t++) {
        // terminal[t] and observation[t] are emitted together after the prior step.
        if (terminals != NULL && to_float(terminals[b * T_seq + t]) != 0.0f) {
            h_t = 0.0f;
        }
        scan_h[h_base + t * H] = from_float(h_t);

        float hidden_val = to_float(__ldg(&combined_h_base[t_offset]));
        float gate_val = to_float(__ldg(&combined_g_base[t_offset]));
        float proj_val = to_float(__ldg(&combined_p_base[t_offset]));
        float x_val = to_float(__ldg(&input[out_base + t * H]));

        // h = (1-z)*h + z*h_tilde  (exact sigmoid; matches prior train log-space)
        float z = sigmoid(gate_val);
        float h_tilde = (hidden_val >= 0.0f) ? hidden_val + 0.5f : sigmoid(hidden_val);
        h_t = lerp(h_t, h_tilde, z);

        float proj_sigmoid = sigmoid(proj_val);
        out[out_curr] = from_float(proj_sigmoid * h_t + (1.0f - proj_sigmoid) * x_val);
        h_t = to_float(from_float(h_t));

        out_curr += H;
        t_offset += H3;
    }

    next_state[bH + h] = from_float(h_t);
}

// Linear reverse scan. Cache is bf16 h only; a,z,h_tilde recomputed from gates in f32.
__global__ void mingru_scan_backward(PrefixScan scan,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ grad_next_state) {
    int T_seq = scan.T, H = scan.H, B = scan.B;
    precision_t* __restrict__ grad_combined = scan.grad_combined.data;
    precision_t* __restrict__ grad_state = scan.grad_state.data;
    precision_t* __restrict__ grad_input = scan.grad_input.data;
    const precision_t* __restrict__ combined = scan.combined_ptr;
    const precision_t* __restrict__ input = scan.input_ptr;
    const precision_t* __restrict__ terminals = scan.terminals_ptr;
    const precision_t* __restrict__ scan_h = scan.scan_h.data;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }

    int b = idx / H;
    int h = idx % H;

    int bHT = b * H * T_seq;
    int cbase = 3 * bHT;
    int H3 = 3 * H;
    int H2 = 2 * H;
    const int state_idx = b * H + h;
    const int out_base = bHT + h;
    const int h_base = b * T_seq * H + h;

    const precision_t* __restrict__ combined_h_base = &combined[cbase + h];
    const precision_t* __restrict__ combined_g_base = &combined[cbase + H + h];
    const precision_t* __restrict__ combined_p_base = &combined[cbase + H2 + h];

    precision_t* __restrict__ grad_combined_h_base = &grad_combined[cbase + h];
    precision_t* __restrict__ grad_combined_g_base = &grad_combined[cbase + H + h];
    precision_t* __restrict__ grad_combined_p_base = &grad_combined[cbase + H2 + h];

    // dh flowing into h_t from the future (and grad_next at t=T).
    float dh = to_float(grad_next_state[state_idx]);

    for (int t = T_seq; t >= 1; --t) {
        int t0 = t - 1;  // 0-based step
        int t_offset = t0 * H3;
        int input_idx = out_base + t0 * H;

        float h_prev = to_float(__ldg(&scan_h[h_base + (t - 1) * H]));

        float hidden_val = to_float(__ldg(&combined_h_base[t_offset]));
        float gate_val = to_float(__ldg(&combined_g_base[t_offset]));
        float proj_val = to_float(__ldg(&combined_p_base[t_offset]));
        float x_val = to_float(__ldg(&input[input_idx]));
        float grad_out_val = to_float(__ldg(&grad_out[input_idx]));

        float z = sigmoid(gate_val);
        float h_tilde = (hidden_val >= 0.0f) ? hidden_val + 0.5f : sigmoid(hidden_val);
        float h_t = lerp(h_prev, h_tilde, z);
        float proj_sigmoid = sigmoid(proj_val);

        // highway: out = s*h + (1-s)*x
        float dh_from_out = grad_out_val * proj_sigmoid;
        float grad_proj = grad_out_val * (h_t - x_val) * proj_sigmoid * (1.0f - proj_sigmoid);
        grad_input[input_idx] = from_float(grad_out_val * (1.0f - proj_sigmoid));
        grad_combined_p_base[t_offset] = from_float(grad_proj);

        float dh_total = dh + dh_from_out;

        // h = (1-z)*h_prev + z*h_tilde
        float d_h_prev = dh_total * (1.0f - z);
        float d_h_tilde = dh_total * z;
        float d_z = dh_total * (h_tilde - h_prev);

        // z = sigmoid(gate)
        float d_gate = d_z * z * (1.0f - z);
        // h_tilde = hidden+0.5  or  sigmoid(hidden)
        float d_hidden = (hidden_val >= 0.0f)
            ? d_h_tilde
            : d_h_tilde * h_tilde * (1.0f - h_tilde);

        grad_combined_h_base[t_offset] = from_float(d_hidden);
        grad_combined_g_base[t_offset] = from_float(d_gate);

        // terminal before this step zeroed h_prev contribution into this step;
        // after bwd through the step, cut gradient flow into pre-reset state.
        dh = d_h_prev;
        if (terminals != NULL &&
                to_float(terminals[b * T_seq + t0]) != 0.0f) {
            dh = 0.0f;
        }
    }

    grad_state[state_idx] = from_float(dh);
}

// Same math as the two scans above, bit for bit, but each thread issues the
// loads for MINGRU_UNROLL steps before the dependent chain. Latency bound at
// B*H <= 32K (default nets): 1.4-1.6x fwd, 1.0-1.45x bwd. Slower once
// bandwidth bound (B*H >= 64K), where the plain kernels stay.
#define MINGRU_UNROLL 8

__global__ void mingru_scan_forward_unroll(PrefixScan scan) {
    int T_seq = scan.T, H = scan.H, B = scan.B;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }
    int b = idx / H, h = idx % H;
    int bHT = b * H * T_seq, H3 = 3 * H;
    const precision_t* __restrict__ comb = scan.combined_ptr + 3 * bHT + h;
    const precision_t* __restrict__ x_in = scan.input_ptr + bHT + h;
    const precision_t* __restrict__ term = scan.terminals_ptr;
    precision_t* __restrict__ out = scan.out.data + bHT + h;
    precision_t* __restrict__ scan_h = scan.scan_h.data + bHT + h;
    float h_t = to_float(scan.state_ptr[b * H + h]);
    for (int t0 = 0; t0 < T_seq; t0 += MINGRU_UNROLL) {
        float hid[MINGRU_UNROLL], gat[MINGRU_UNROLL], prj[MINGRU_UNROLL];
        float xv[MINGRU_UNROLL], tm[MINGRU_UNROLL];
        #pragma unroll
        for (int i = 0; i < MINGRU_UNROLL; i++) {
            int t = t0 + i;
            hid[i] = to_float(__ldg(&comb[t * H3]));
            gat[i] = to_float(__ldg(&comb[t * H3 + H]));
            prj[i] = to_float(__ldg(&comb[t * H3 + 2 * H]));
            xv[i] = to_float(__ldg(&x_in[t * H]));
            tm[i] = term != NULL ? to_float(term[b * T_seq + t]) : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < MINGRU_UNROLL; i++) {
            int t = t0 + i;
            if (tm[i] != 0.0f) {
                h_t = 0.0f;
            }
            scan_h[t * H] = from_float(h_t);
            float z = sigmoid(gat[i]);
            float h_tilde = (hid[i] >= 0.0f) ? hid[i] + 0.5f : sigmoid(hid[i]);
            h_t = lerp(h_t, h_tilde, z);
            float s = sigmoid(prj[i]);
            out[t * H] = from_float(s * h_t + (1.0f - s) * xv[i]);
            h_t = to_float(from_float(h_t));
        }
    }
    scan.next_state.data[b * H + h] = from_float(h_t);
}

__global__ void mingru_scan_backward_unroll(PrefixScan scan,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ grad_next_state) {
    int T_seq = scan.T, H = scan.H, B = scan.B;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * H) {
        return;
    }
    int b = idx / H, h = idx % H;
    int bHT = b * H * T_seq, H3 = 3 * H;
    const precision_t* __restrict__ comb = scan.combined_ptr + 3 * bHT + h;
    precision_t* __restrict__ gcomb = scan.grad_combined.data + 3 * bHT + h;
    const precision_t* __restrict__ x_in = scan.input_ptr + bHT + h;
    const precision_t* __restrict__ scan_h = scan.scan_h.data + bHT + h;
    const precision_t* __restrict__ gout = grad_out + bHT + h;
    precision_t* __restrict__ gin = scan.grad_input.data + bHT + h;
    const precision_t* __restrict__ term = scan.terminals_ptr;
    float dh = to_float(grad_next_state[b * H + h]);
    for (int t0 = T_seq - MINGRU_UNROLL; t0 >= 0; t0 -= MINGRU_UNROLL) {
        float hp[MINGRU_UNROLL], hid[MINGRU_UNROLL], gat[MINGRU_UNROLL];
        float prj[MINGRU_UNROLL], xv[MINGRU_UNROLL], go[MINGRU_UNROLL];
        float tm[MINGRU_UNROLL];
        #pragma unroll
        for (int i = 0; i < MINGRU_UNROLL; i++) {
            int t = t0 + i;
            hp[i] = to_float(__ldg(&scan_h[t * H]));
            hid[i] = to_float(__ldg(&comb[t * H3]));
            gat[i] = to_float(__ldg(&comb[t * H3 + H]));
            prj[i] = to_float(__ldg(&comb[t * H3 + 2 * H]));
            xv[i] = to_float(__ldg(&x_in[t * H]));
            go[i] = to_float(__ldg(&gout[t * H]));
            tm[i] = term != NULL ? to_float(term[b * T_seq + t]) : 0.0f;
        }
        #pragma unroll
        for (int i = MINGRU_UNROLL - 1; i >= 0; i--) {
            int t = t0 + i;
            float z = sigmoid(gat[i]);
            float h_tilde = (hid[i] >= 0.0f) ? hid[i] + 0.5f : sigmoid(hid[i]);
            float h_t = lerp(hp[i], h_tilde, z);
            float proj_sigmoid = sigmoid(prj[i]);
            float dh_from_out = go[i] * proj_sigmoid;
            float grad_proj = go[i] * (h_t - xv[i]) * proj_sigmoid * (1.0f - proj_sigmoid);
            gin[t * H] = from_float(go[i] * (1.0f - proj_sigmoid));
            gcomb[t * H3 + 2 * H] = from_float(grad_proj);
            float dh_total = dh + dh_from_out;
            float d_h_prev = dh_total * (1.0f - z);
            float d_h_tilde = dh_total * z;
            float d_z = dh_total * (h_tilde - hp[i]);
            float d_gate = d_z * z * (1.0f - z);
            float d_hidden = (hid[i] >= 0.0f)
                ? d_h_tilde
                : d_h_tilde * h_tilde * (1.0f - h_tilde);
            gcomb[t * H3] = from_float(d_hidden);
            gcomb[t * H3 + H] = from_float(d_gate);
            dh = d_h_prev;
            if (tm[i] != 0.0f) {
                dh = 0.0f;
            }
        }
    }
    scan.grad_state.data[b * H + h] = from_float(dh);
}

// Coalesced T-tile load + sequential apply. 32 h-threads apply after the load.
// Use when T is long and seq occupancy B*H is not already huge.
#define MINGRU_TILE 32

__global__ void mingru_scan_forward_row(PrefixScan scan) {
    int T_seq = scan.T, H = scan.H;
    int hx = threadIdx.x, t_local = threadIdx.y;
    int h = blockIdx.y * MINGRU_TILE + hx;
    int h_ok = h < H;
    int b = blockIdx.x, H3 = 3 * H, bHT = b * H * T_seq, cbase = 3 * bHT;
    float h_t = h_ok ? to_float(scan.state_ptr[b * H + h]) : 0.0f;
    __shared__ float sm_z[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_ht[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_s[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_x[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_term[MINGRU_TILE][MINGRU_TILE];
    for (int c = 0; c < (T_seq + MINGRU_TILE - 1) / MINGRU_TILE; c++) {
        int t = c * MINGRU_TILE + t_local;
        float z = 0.0f, h_tilde = 0.0f, s = 0.0f, x = 0.0f, term = 0.0f;
        if (t < T_seq && h_ok) {
            int off = t * H3 + h;
            float hidden = to_float(scan.combined_ptr[cbase + off]);
            float gate = to_float(scan.combined_ptr[cbase + H + off]);
            float proj = to_float(scan.combined_ptr[cbase + 2 * H + off]);
            x = to_float(scan.input_ptr[bHT + t * H + h]);
            z = sigmoid(gate);
            h_tilde = (hidden >= 0.0f) ? hidden + 0.5f : sigmoid(hidden);
            s = sigmoid(proj);
            term = scan.terminals_ptr != NULL &&
                to_float(scan.terminals_ptr[b * T_seq + t]) != 0.0f;
        }
        sm_z[t_local][hx] = z;
        sm_ht[t_local][hx] = h_tilde;
        sm_s[t_local][hx] = s;
        sm_x[t_local][hx] = x;
        sm_term[t_local][hx] = term;
        __syncthreads();
        if (t_local == 0 && h_ok) {
            int nloc = min(MINGRU_TILE, T_seq - c * MINGRU_TILE);
            for (int i = 0; i < nloc; i++) {
                if (sm_term[i][hx] != 0.0f) {
                    h_t = 0.0f;
                }
                int tt = c * MINGRU_TILE + i;
                scan.scan_h.data[bHT + tt * H + h] = from_float(h_t);
                h_t = lerp(h_t, sm_ht[i][hx], sm_z[i][hx]);
                float sv = sm_s[i][hx];
                scan.out.data[bHT + tt * H + h] =
                    from_float(sv * h_t + (1.0f - sv) * sm_x[i][hx]);
                h_t = to_float(from_float(h_t));
            }
        }
        __syncthreads();
    }
    if (t_local == 0 && h_ok) {
        scan.next_state.data[b * H + h] = from_float(h_t);
    }
}

__global__ void mingru_scan_backward_row(PrefixScan scan,
        const precision_t* __restrict__ grad_out,
        const precision_t* __restrict__ grad_next_state) {
    int T_seq = scan.T, H = scan.H;
    int hx = threadIdx.x, t_local = threadIdx.y;
    int h = blockIdx.y * MINGRU_TILE + hx;
    int h_ok = h < H;
    int b = blockIdx.x, H3 = 3 * H, bHT = b * H * T_seq, cbase = 3 * bHT;
    float dh = h_ok ? to_float(grad_next_state[b * H + h]) : 0.0f;
    __shared__ float sm_z[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_ht[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_s[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_x[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_hp[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_go[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_hid[MINGRU_TILE][MINGRU_TILE];
    __shared__ float sm_term[MINGRU_TILE][MINGRU_TILE];
    for (int c = (T_seq + MINGRU_TILE - 1) / MINGRU_TILE - 1; c >= 0; c--) {
        int t = c * MINGRU_TILE + t_local;
        float z = 0.0f, h_tilde = 0.0f, s = 0.0f, x = 0.0f;
        float h_prev = 0.0f, gout = 0.0f, hidden = 0.0f, term = 0.0f;
        if (t < T_seq && h_ok) {
            int off = t * H3 + h;
            hidden = to_float(scan.combined_ptr[cbase + off]);
            float gate = to_float(scan.combined_ptr[cbase + H + off]);
            float proj = to_float(scan.combined_ptr[cbase + 2 * H + off]);
            x = to_float(scan.input_ptr[bHT + t * H + h]);
            h_prev = to_float(scan.scan_h.data[bHT + t * H + h]);
            gout = to_float(grad_out[bHT + t * H + h]);
            z = sigmoid(gate);
            h_tilde = (hidden >= 0.0f) ? hidden + 0.5f : sigmoid(hidden);
            s = sigmoid(proj);
            term = scan.terminals_ptr != NULL &&
                to_float(scan.terminals_ptr[b * T_seq + t]) != 0.0f;
        }
        sm_z[t_local][hx] = z;
        sm_ht[t_local][hx] = h_tilde;
        sm_s[t_local][hx] = s;
        sm_x[t_local][hx] = x;
        sm_hp[t_local][hx] = h_prev;
        sm_go[t_local][hx] = gout;
        sm_hid[t_local][hx] = hidden;
        sm_term[t_local][hx] = term;
        __syncthreads();
        if (t_local == 0 && h_ok) {
            for (int i = min(MINGRU_TILE, T_seq - c * MINGRU_TILE) - 1; i >= 0; i--) {
                int tt = c * MINGRU_TILE + i;
                z = sm_z[i][hx];
                s = sm_s[i][hx];
                gout = sm_go[i][hx];
                h_tilde = sm_ht[i][hx];
                h_prev = sm_hp[i][hx];
                float h_t = lerp(h_prev, h_tilde, z);
                float dh_total = dh + gout * s;
                float d_h_tilde = dh_total * z;
                int off = tt * H3 + h;
                scan.grad_input.data[bHT + tt * H + h] =
                    from_float(gout * (1.0f - s));
                scan.grad_combined.data[cbase + off] = from_float(
                    sm_hid[i][hx] >= 0.0f ? d_h_tilde
                    : d_h_tilde * h_tilde * (1.0f - h_tilde));
                scan.grad_combined.data[cbase + H + off] =
                    from_float(dh_total * (h_tilde - h_prev) * z * (1.0f - z));
                scan.grad_combined.data[cbase + 2 * H + off] =
                    from_float(gout * (h_t - sm_x[i][hx]) * s * (1.0f - s));
                dh = sm_term[i][hx] != 0.0f ? 0.0f : dh_total * (1.0f - z);
            }
        }
        __syncthreads();
    }
    if (t_local == 0 && h_ok) {
        scan.grad_state.data[b * H + h] = from_float(dh);
    }
}

#if !defined(PRECISION_FLOAT)
// Fused MinGRU training layer for H == 32 (bf16, sm_80+). A block owns MGF_ROWS
// consecutive (b, t) rows = whole sequences (T divides MGF_ROWS); warp w owns the
// MGF_U-row segment MGF_U*w .. MGF_U*w + MGF_U - 1 of one sequence. The layer GEMM
// runs on tensor cores into shared memory, so the (rows, 3H) combined buffer never
// reaches global memory: backward recomputes it from x and reads the forward's bf16
// h_{t-1} per step. Small nets are latency bound with few scan chains per SM, so both
// directions split each sequence's chain over its warps' segments. Pass A: the
// segment's linear chain from a zero carry, publishing (local end value, keep
// product). Pass B: fold the neighbouring segments into the carry, rerun the chain
// with the per-step math. Blocks are small enough for four per SM. dX and a
// per-block dW partial (summed in a fixed order by mingru_fused_dw_reduce) run on
// tensor cores. One fwd kernel replaces GEMM + scan; one bwd kernel replaces scan +
// dX / dW GEMMs + split-K reduce.
#define MGF_H 32
#define MGF_U 8                  // rows per warp segment (T % MGF_U == 0)
#define MGF_WARPS 8
#define MGF_ROWS (MGF_WARPS * MGF_U)
#define MGF_XLD (MGF_H + 8)      // bf16 row strides: fragment loads hit distinct banks
#define MGF_CLD (3 * MGF_H + 8)
#define MGF_XS_BYTES (MGF_ROWS * MGF_XLD * 2)
#define MGF_CS_BYTES (MGF_ROWS * MGF_CLD * 2)
#define MGF_LP_BYTES (MGF_WARPS * 32 * 8)  // per-lane (local end value, keep product)
// Backward adds Go (grad_out -> highway grad; unpadded, only the dX init reads it by
// fragment): 24 KB, exactly four blocks in the 100 KB of an sm_86 SM.
#define MGF_GO_BYTES (MGF_ROWS * MGF_H * 2)
#define MGF_FWD_SMEM (MGF_XS_BYTES + MGF_CS_BYTES + MGF_LP_BYTES)
#define MGF_BWD_SMEM (MGF_XS_BYTES + MGF_CS_BYTES + MGF_GO_BYTES + MGF_LP_BYTES)
#define MGF_RED_WARPS 8

// Fast-math gates, shared by the rollout step, forward and backward: the train replay
// and the backward see exactly the gates the rollout applied.
__device__ __forceinline__ float mgf_sigmoid(float x) {
    float e = __expf(-fabsf(x));
    float r = __fdividef(1.0f, 1.0f + e);
    return x >= 0.0f ? r : e * r;
}

__device__ __forceinline__ float mgf_htilde(float hid) {
    return hid >= 0.0f ? hid + 0.5f : mgf_sigmoid(hid);
}

// dst (row stride LD) <- MGF_ROWS rows of src (row stride MGF_H), 16 B per step.
template <int LD>
__device__ __forceinline__ void mgf_stage_rows(precision_t* dst,
        const precision_t* __restrict__ src) {
    const uint4* s = (const uint4*)src;
    for (int i = threadIdx.x; i < MGF_ROWS * MGF_H / 8; i += blockDim.x) {
        *(uint4*)&dst[i / (MGF_H / 8) * LD + 8 * (i % (MGF_H / 8))] = s[i];
    }
}

// Reset bits of the MGF_U rows from g0: one 16 B load (g0 % MGF_U == 0).
__device__ __forceinline__ unsigned int mgf_resets(
        const precision_t* __restrict__ terminals, long g0) {
    unsigned int reset = 0;
    if (terminals != NULL) {
        uint4 tv = *(const uint4*)&terminals[g0];
        const precision_t* tp = (const precision_t*)&tv;
        #pragma unroll
        for (int u = 0; u < MGF_U; u++) {
            reset |= (unsigned int)(to_float(tp[u]) != 0.0f) << u;
        }
    }
    return reset;
}

__device__ __forceinline__ unsigned int mgf_smem_addr(const void* p) {
    return (unsigned int)__cvta_generic_to_shared(p);
}

// ldmatrix: lane l addresses row l % 8 of 8x8 bf16 tile l / 8 (x2: lanes 0-15);
// register i holds tile i in mma fragment layout (.trans: the transposed tile).
__device__ __forceinline__ void ldsm_x4(unsigned int* r, const void* p) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(mgf_smem_addr(p)) : "memory");
#endif
}

__device__ __forceinline__ void ldsm_x4_trans(unsigned int* r, const void* p) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(mgf_smem_addr(p)) : "memory");
#endif
}

__device__ __forceinline__ void ldsm_x2_trans(unsigned int* r, const void* p) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
        : "=r"(r[0]), "=r"(r[1]) : "r"(mgf_smem_addr(p)) : "memory");
#endif
}

// Cs = bf16(Xs @ W^T), W (3H, H) row-major: b fragment (k, n) = W[n][k] is one
// 32-bit load. Task = (16-row m-tile, half of the 3H output columns).
__device__ __forceinline__ void mgf_combined(precision_t* Cs, const precision_t* Xs,
        const precision_t* __restrict__ W) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    int q = lane / 8, rr = lane % 8;
    const unsigned int* Wu = (const unsigned int*)W;
    for (int task = warp; task < MGF_ROWS / 16 * 2; task += MGF_WARPS) {
        int r0 = 16 * (task / 2), j0 = task % 2 * (3 * MGF_H / 16);
        unsigned int a[MGF_H / 16][4];
        #pragma unroll
        for (int kk = 0; kk < MGF_H / 16; kk++) {
            ldsm_x4(a[kk], &Xs[(r0 + q % 2 * 8 + rr) * MGF_XLD + 16 * kk + q / 2 * 8]);
        }
        #pragma unroll
        for (int jj = 0; jj < 3 * MGF_H / 16; jj++) {
            int j = j0 + jj;
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int kk = 0; kk < MGF_H / 16; kk++) {
                int w = ((8 * j + g) * MGF_H + 16 * kk + 2 * t) / 2;
                mma_16816(d, a[kk], __ldg(&Wu[w]), __ldg(&Wu[w + 4]));
            }
            int c = 8 * j + 2 * t;
            *(__nv_bfloat162*)&Cs[(r0 + g) * MGF_CLD + c] = __floats2bfloat162_rn(d[0], d[1]);
            *(__nv_bfloat162*)&Cs[(r0 + g + 8) * MGF_CLD + c] =
                __floats2bfloat162_rn(d[2], d[3]);
        }
    }
}

// All L layers of the train forward in one launch: a block's rows stay in shared
// memory between layers (pass B writes layer l's out over its input, layer l + 1's
// x). Per layer: out[l] and hs[l] (the backward's inputs; hs = each step's post-reset
// recurrent input h_{t-1}) and wt[l] = W^T (H, 3H), one coalesced element per thread
// across the grid, so the backward's dX b fragments are single 32-bit loads.
// h_t = keep*h_{t-1} + z*h~, keep = 0 at a reset, else 1 - z. Pass B rounds h to bf16
// after every step like the rollout state; a segment's carry comes from the unrounded
// fold (within ~1 ulp).
#define MGF_MAX_LAYERS 16
struct MgfFwdLayers {
    const precision_t* W[MGF_MAX_LAYERS];
    const precision_t* state[MGF_MAX_LAYERS];
    precision_t* wt[MGF_MAX_LAYERS];
    precision_t* out[MGF_MAX_LAYERS];
    precision_t* hs[MGF_MAX_LAYERS];
};

__global__ void __launch_bounds__(MGF_WARPS * 32, 4) mingru_fused_fwd(MgfFwdLayers p,
        int L, const precision_t* __restrict__ x, const precision_t* __restrict__ terminals,
        int T) {
    extern __shared__ __align__(16) unsigned char mgf_smem[];
    precision_t* Xs = (precision_t*)mgf_smem;
    precision_t* Cs = (precision_t*)(mgf_smem + MGF_XS_BYTES);
    float2* LP = (float2*)(mgf_smem + MGF_XS_BYTES + MGF_CS_BYTES);
    long row0 = (long)blockIdx.x * MGF_ROWS;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int l = 0; l < L; l++) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < 3 * MGF_H * MGF_H;
                i += gridDim.x * blockDim.x) {
            p.wt[l][i] = p.W[l][i % (3 * MGF_H) * MGF_H + i / (3 * MGF_H)];
        }
    }
    int s0 = warp * MGF_U, seg = s0 % T / MGF_U;
    long g0 = row0 + s0;
    unsigned int reset = mgf_resets(terminals, g0);
    mgf_stage_rows<MGF_XLD>(Xs, x + row0 * MGF_H);
    for (int l = 0; l < L; l++) {
        float hc = to_float(p.state[l][g0 / T * MGF_H + lane]);
        __syncthreads();
        mgf_combined(Cs, Xs, p.W[l]);
        __syncthreads();

        // Pass A: the segment's gates, its end h for a zero carry and its keep product.
        float z[MGF_U], ht[MGF_U];
        float h = 0.0f, prod = 1.0f;
        #pragma unroll
        for (int u = 0; u < MGF_U; u++) {
            const precision_t* c = &Cs[(s0 + u) * MGF_CLD];
            ht[u] = mgf_htilde(to_float(c[lane]));
            z[u] = mgf_sigmoid(to_float(c[MGF_H + lane]));
            float keep = reset >> u & 1 ? 0.0f : 1.0f - z[u];
            h = keep * h + z[u] * ht[u];
            prod *= keep;
        }
        LP[threadIdx.x] = make_float2(h, prod);
        __syncthreads();

        // Pass B: carry = the state folded through the sequence's earlier segments.
        for (int w = warp - seg; w < warp; w++) {
            float2 lp = LP[w * 32 + lane];
            hc = lp.x + lp.y * hc;
        }
        hc = to_float(from_float(hc));
        #pragma unroll
        for (int u = 0; u < MGF_U; u++) {
            int r = s0 + u;
            long i = (row0 + r) * MGF_H + lane;
            hc = reset >> u & 1 ? 0.0f : hc;
            p.hs[l][i] = from_float(hc);
            hc = lerp(hc, ht[u], z[u]);
            float sp = mgf_sigmoid(to_float(Cs[r * MGF_CLD + 2 * MGF_H + lane]));
            precision_t* xr = &Xs[r * MGF_XLD + lane];
            precision_t o = from_float(sp * hc + (1.0f - sp) * to_float(*xr));
            p.out[l][i] = o;
            *xr = o;
            hc = to_float(from_float(hc));
        }
    }
}

// All L layers of the backward in one launch, top layer first. A block's rows stay
// in shared memory: layer l's dX (written over its highway grad in Go) is layer
// l - 1's grad_out, and layer l - 1's input rows and h_{t-1} load into registers while
// layer l computes dX / dW. Layer 0's dX goes to grad_in. dW partials:
// dw_part[l][blockIdx.x], summed by mingru_fused_dw_reduce.
struct MgfBwdLayers {
    const precision_t* x[MGF_MAX_LAYERS];   // layer inputs (encoder out, prior outs)
    const precision_t* W[MGF_MAX_LAYERS];
    const precision_t* Wt[MGF_MAX_LAYERS];
    const precision_t* hs[MGF_MAX_LAYERS];
};

__global__ void __launch_bounds__(MGF_WARPS * 32, 4) mingru_fused_bwd(MgfBwdLayers p,
        int L, precision_t* __restrict__ grad_in, float* __restrict__ dw_part,
        const precision_t* __restrict__ terminals, const precision_t* __restrict__ grad_out,
        int T) {
    extern __shared__ __align__(16) unsigned char mgf_smem[];
    precision_t* Xs = (precision_t*)mgf_smem;
    precision_t* Cs = (precision_t*)(mgf_smem + MGF_XS_BYTES);
    precision_t* Go = (precision_t*)(mgf_smem + MGF_XS_BYTES + MGF_CS_BYTES);
    float2* LP = (float2*)(mgf_smem + MGF_XS_BYTES + MGF_CS_BYTES + MGF_GO_BYTES);
    long row0 = (long)blockIdx.x * MGF_ROWS;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int g = lane / 4, t4 = lane % 4, q = lane / 8, rr = lane % 8;

    // The segment's resets and the forward's h_{t-1} load first, under the staging
    // and the combined GEMM.
    int s0 = warp * MGF_U;
    long g0 = row0 + s0;
    unsigned int reset = mgf_resets(terminals, g0);
    float hp[MGF_U];
    #pragma unroll
    for (int u = 0; u < MGF_U; u++) {
        hp[u] = to_float(p.hs[L - 1][(g0 + u) * MGF_H + lane]);
    }
    mgf_stage_rows<MGF_XLD>(Xs, p.x[L - 1] + row0 * MGF_H);
    mgf_stage_rows<MGF_H>(Go, grad_out + row0 * MGF_H);

    for (int l = L - 1; l >= 0; l--) {
        __syncthreads();
        mgf_combined(Cs, Xs, p.W[l]);
        __syncthreads();

        // dh chain (lane = h), reverse in t: dht = dh + go*s, dh <- dht*keep, keep = 0
        // at a reset, else 1 - z. grad_next_state is never written in training: dh
        // enters each sequence's last step as zero. Pass A: the segment's dh out for a
        // zero carry-in and its keep product, so dh out = local + prod * carry.
        float z[MGF_U], sg[MGF_U];
        float dh = 0.0f, prod = 1.0f;
        #pragma unroll
        for (int u = MGF_U - 1; u >= 0; u--) {
            const precision_t* c = &Cs[(s0 + u) * MGF_CLD];
            z[u] = mgf_sigmoid(to_float(c[MGF_H + lane]));
            sg[u] = mgf_sigmoid(to_float(c[2 * MGF_H + lane]));
            float keep = reset >> u & 1 ? 0.0f : 1.0f - z[u];
            dh = (dh + to_float(Go[(s0 + u) * MGF_H + lane]) * sg[u]) * keep;
            prod *= keep;
        }
        LP[threadIdx.x] = make_float2(dh, prod);
        __syncthreads();

        // Pass B: fold the later segments of the sequence into the carry, rerun the
        // chain and write the per-step grads: G in place over combined, highway into Go.
        dh = 0.0f;
        for (int w = (s0 / T + 1) * T / MGF_U - 1; w > warp; w--) {
            float2 lp = LP[w * 32 + lane];
            dh = lp.x + lp.y * dh;
        }
        #pragma unroll
        for (int u = MGF_U - 1; u >= 0; u--) {
            int r = s0 + u;
            precision_t* c = &Cs[r * MGF_CLD];
            float hid = to_float(c[lane]);
            float ht = mgf_htilde(hid), zz = z[u], s = sg[u], h_prev = hp[u];
            float go = to_float(Go[r * MGF_H + lane]);
            float dht = dh + go * s;
            Go[r * MGF_H + lane] = from_float(go * (1.0f - s));
            c[lane] = from_float(dht * (hid >= 0.0f ? zz : zz * ht * (1.0f - ht)));
            c[MGF_H + lane] = from_float(dht * (ht - h_prev) * zz * (1.0f - zz));
            c[2 * MGF_H + lane] = from_float(go * (lerp(h_prev, ht, zz)
                - to_float(Xs[r * MGF_XLD + lane])) * s * (1.0f - s));
            dh = reset >> u & 1 ? 0.0f : dht * (1.0f - zz);
        }
        __syncthreads();

        // Layer l - 1's input rows (one 16 B chunk per thread) load during dX / dW.
        uint4 x_next = make_uint4(0, 0, 0, 0);
        if (l > 0) {
            x_next = ((const uint4*)(p.x[l - 1] + row0 * MGF_H))[threadIdx.x];
        }

        // dX = G @ W + highway, over the highway grad in Go (each tile reads only its
        // own Go entries): b fragment (k = j, n = f) = Wt[f][j], Wt[f][j + 1].
        precision_t* dx = l > 0 ? Go : grad_in + row0 * MGF_H;
        const unsigned int* Wtu = (const unsigned int*)p.Wt[l];
        for (int task = warp; task < MGF_ROWS / 16 * (MGF_H / 8); task += MGF_WARPS) {
            int m0 = 16 * (task / (MGF_H / 8)), f0 = 8 * (task % (MGF_H / 8));
            int f = f0 + 2 * t4, n = f0 + g;
            float2 h0 = __bfloat1622float2(
                *(const __nv_bfloat162*)&Go[(m0 + g) * MGF_H + f]);
            float2 h1 = __bfloat1622float2(
                *(const __nv_bfloat162*)&Go[(m0 + g + 8) * MGF_H + f]);
            float d[4] = {h0.x, h0.y, h1.x, h1.y};
            #pragma unroll
            for (int kk = 0; kk < 3 * MGF_H / 16; kk++) {
                unsigned int a[4];
                ldsm_x4(a, &Cs[(m0 + q % 2 * 8 + rr) * MGF_CLD + 16 * kk + q / 2 * 8]);
                int w = (n * 3 * MGF_H + 16 * kk + 2 * t4) / 2;
                mma_16816(d, a, __ldg(&Wtu[w]), __ldg(&Wtu[w + 4]));
            }
            *(__nv_bfloat162*)&dx[(m0 + g) * MGF_H + f] = __floats2bfloat162_rn(d[0], d[1]);
            *(__nv_bfloat162*)&dx[(m0 + g + 8) * MGF_H + f] =
                __floats2bfloat162_rn(d[2], d[3]);
        }

        // dW partial (3H, H) = G^T @ X over the block's rows: a (m = j, k = row) and
        // b (k = row, n = f) fragments are transposed ldmatrix tiles of Cs and Xs.
        float* part = dw_part + ((long)l * gridDim.x + blockIdx.x) * 3 * MGF_H * MGF_H;
        for (int tile = warp; tile < 3 * MGF_H / 16 * (MGF_H / 8); tile += MGF_WARPS) {
            int j0 = 16 * (tile / (MGF_H / 8)), f0 = 8 * (tile % (MGF_H / 8));
            float d[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int kr = 0; kr < MGF_ROWS; kr += 16) {
                unsigned int a[4], b[2];
                ldsm_x4_trans(a, &Cs[(kr + q / 2 * 8 + rr) * MGF_CLD + j0 + q % 2 * 8]);
                ldsm_x2_trans(b, &Xs[(kr + q % 2 * 8 + rr) * MGF_XLD + f0]);
                mma_16816(d, a, b[0], b[1]);
            }
            *(float2*)&part[(j0 + g) * MGF_H + f0 + 2 * t4] = make_float2(d[0], d[1]);
            *(float2*)&part[(j0 + g + 8) * MGF_H + f0 + 2 * t4] = make_float2(d[2], d[3]);
        }

        if (l > 0) {
            #pragma unroll
            for (int u = 0; u < MGF_U; u++) {
                hp[u] = to_float(p.hs[l - 1][(g0 + u) * MGF_H + lane]);
            }
            __syncthreads();
            *(uint4*)&Xs[threadIdx.x / (MGF_H / 8) * MGF_XLD + 8 * (threadIdx.x % (MGF_H / 8))] =
                x_next;
        }
    }
}

// out.p[l] = sum over blocks of layer l's dW partials (blockIdx.y = l), on the
// background stream after the backward, under the encoder backward. Block = 32
// outputs: warp w sums partials w, w + MGF_RED_WARPS, ... (128 B rows), then a
// fixed-order sum over warps.
struct MgfWgrad {
    precision_t* p[MGF_MAX_LAYERS];
};

__global__ void __launch_bounds__(MGF_RED_WARPS * 32) mingru_fused_dw_reduce(
        MgfWgrad out, const float* __restrict__ part, int nblk) {
    __shared__ float acc[MGF_RED_WARPS][32];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, i = blockIdx.x * 32 + lane;
    const float* src = part + (long)blockIdx.y * nblk * 3 * MGF_H * MGF_H + i;
    float s = 0.0f;
    #pragma unroll 8
    for (int b = warp; b < nblk; b += MGF_RED_WARPS) {
        s += src[(long)b * 3 * MGF_H * MGF_H];
    }
    acc[warp][lane] = s;
    __syncthreads();
    if (warp == 0) {
        float t = 0.0f;
        #pragma unroll
        for (int w = 0; w < MGF_RED_WARPS; w++) {
            t += acc[w][lane];
        }
        out.p[blockIdx.y][i] = from_float(t);
    }
}

// Rollout step through all L layers for B agents, state (L, B, H) updated in place:
// block = MGF_ROWS agents, x stays in shared memory between layers. Gate math and h
// rounding match mingru_fused_fwd.
#define MGF_STEP_SMEM (MGF_XS_BYTES + MGF_CS_BYTES)
struct MgfLayers {
    const precision_t* W[MGF_MAX_LAYERS];
};

__global__ void __launch_bounds__(MGF_WARPS * 32) mingru_fused_step(
        precision_t* __restrict__ out, precision_t* __restrict__ state,
        const precision_t* __restrict__ x, MgfLayers w, int L, int B) {
    extern __shared__ __align__(16) unsigned char mgf_smem[];
    precision_t* Xs = (precision_t*)mgf_smem;
    precision_t* Cs = (precision_t*)(mgf_smem + MGF_XS_BYTES);
    long row0 = (long)blockIdx.x * MGF_ROWS;
    int rows = (int)min((long)MGF_ROWS, B - row0);
    for (int i = threadIdx.x; i < MGF_ROWS * MGF_H / 8; i += blockDim.x) {
        int r = i / (MGF_H / 8);
        uint4 v = r < rows ? ((const uint4*)(x + row0 * MGF_H))[i] : make_uint4(0, 0, 0, 0);
        *(uint4*)&Xs[r * MGF_XLD + 8 * (i % (MGF_H / 8))] = v;
    }
    for (int l = 0; l < L; l++) {
        __syncthreads();
        mgf_combined(Cs, Xs, w.W[l]);
        __syncthreads();
        precision_t* st = state + ((long)l * B + row0) * MGF_H;
        for (int i = threadIdx.x; i < rows * MGF_H; i += blockDim.x) {
            int r = i / MGF_H, h = i % MGF_H;
            const precision_t* c = &Cs[r * MGF_CLD];
            float z = mgf_sigmoid(to_float(c[MGF_H + h]));
            float s = mgf_sigmoid(to_float(c[2 * MGF_H + h]));
            float hn = lerp(to_float(st[i]), mgf_htilde(to_float(c[h])), z);
            st[i] = from_float(hn);
            precision_t* xr = &Xs[r * MGF_XLD + h];
            *xr = from_float(s * hn + (1.0f - s) * to_float(*xr));
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < rows * MGF_H / 8; i += blockDim.x) {
        ((uint4*)(out + row0 * MGF_H))[i] =
            *(const uint4*)&Xs[i / (MGF_H / 8) * MGF_XLD + 8 * (i % (MGF_H / 8))];
    }
}
#endif

// One block per column: deterministic tree sum over R rows (R=8192: ~7us; one
// thread per column took ~124us).
__global__ void sum_rows_to_precision_kernel(precision_t* __restrict__ dst,
        const float* __restrict__ src, int R, int C) {
    __shared__ float sdata[256];
    __shared__ float total;
    int col = blockIdx.x, tid = threadIdx.x;
    float sum = 0.0f;
    for (int r = tid; r < R; r += blockDim.x) {
        sum += src[r * C + col];
    }
    sdata[tid] = sum;
    block_reduce_sum(sdata, &total, tid, blockDim.x, 1);
    if (tid == 0) {
        dst[col] = from_float(total);
    }
}

__global__ void assemble_decoder_grad(
        precision_t* __restrict__ dst, const float* __restrict__ grad_logits,
        const float* __restrict__ grad_value, int B_TT, int od, int od_plus_1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B_TT * od_plus_1) {
        return;
    }
    int row = idx / od_plus_1, col = idx % od_plus_1;
    dst[idx] = from_float((col < od) ? grad_logits[row * od + col] : grad_value[row]);
}

Prec encoder_forward(void* w, void* activations, Prec input, cudaStream_t stream) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    // Train acts own wgrad_scratch. The input outlives backward: keep a view for dW.
    if (a->wgrad_scratch.data) {
        a->saved_input = input;
    }
    puf_mm(&input, &ew->weight, &a->out, stream);
    return a->out;
}

void encoder_backward(void* w, void* activations, Prec grad, cudaStream_t stream) {
    EncoderActivations* a = (EncoderActivations*)activations;
    // Last layer before join: nothing left on main to overlap, so sync dW.
    puf_mm_tn(&grad, &a->saved_input, &a->wgrad_scratch, stream);
}

void encoder_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    EncoderWeights* ew = (EncoderWeights*)w;
    Prec wt = {
        .data = ew->weight.data,
        .shape = {ew->out_dim, ew->in_dim},
    };
    puf_kaiming_init(&wt, sqrtf(2.0f), (*seed)++, stream);
}

void encoder_reg_params(void* w, Allocator* alloc) {
    EncoderWeights* ew = (EncoderWeights*)w;
    ew->weight = {.shape = {ew->out_dim, ew->in_dim}};
    alloc_register(alloc, &ew->weight);
}

void encoder_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    *a = (EncoderActivations){
        .out =              {.shape = {B_TT, ew->out_dim}},
        .wgrad_scratch =    {.shape = {ew->out_dim, ew->in_dim}},
    };
    alloc_register(acts, &a->out);
    alloc_register(grads, &a->wgrad_scratch);
}

void encoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    a->out = {.shape = {B, ew->out_dim}};
    alloc_register(alloc, &a->out);
}

void* encoder_create_weights(void* self) {
    Encoder* e = (Encoder*)self;
    EncoderWeights* ew = (EncoderWeights*)calloc(1, sizeof(EncoderWeights));
    ew->in_dim = e->in_dim; ew->out_dim = e->out_dim;
    return ew;
}

#ifdef OBS_PACKED_FEATURES
// Packed obs (env defines OBS_PACKED_*): OBS_PACKED_SCALARS bf16 features, then
// OBS_PACKED_WORDS raw uint32 words whose bit b is binary feature SCALARS + b.
// Same weights (H, OBS_PACKED_FEATURES) as the dense encoder; the kernels build
// mma.m16n8k16 fragments straight from packed rows staged in shared memory, so
// rollout copies, the train transpose and both encoder GEMMs move 2 * OBS_SIZE
// bytes per row instead of 2 * OBS_PACKED_FEATURES.
#define PACKED_K 128          // features padded to 8 k-steps of 16
#define PACKED_N 32           // outputs per block (blockIdx.y): W / grad column slice
#define PACKED_WARPS 4
#define PACKED_ROW_WORDS (OBS_SIZE / 2)
#define PACKED_DW_ROWS 64     // dW rows staged per pass
#define PACKED_DW_BLOCKS 128  // dW row chunks, summed in a fixed order
#define PACKED_DW_WARPS 8
// dW tile stride in row pairs (32 + 4): fragment loads hit 32 distinct banks.
#define PACKED_TLD (PACKED_DW_ROWS / 2 + 4)
// Words of packed rows / grad pairs each thread stages per fwd tile / dW pass.
#define PACKED_FWD_WORDS ((16 * PACKED_ROW_WORDS + 31) / 32)
#define PACKED_DW_WORDS ((PACKED_DW_ROWS * PACKED_ROW_WORDS + PACKED_DW_WARPS * 32 - 1) \
    / (PACKED_DW_WARPS * 32))
#define PACKED_DW_GRAD (PACKED_N * PACKED_DW_ROWS / 4 / (PACKED_DW_WARPS * 32))
static_assert(PACKED_N * PACKED_DW_ROWS / 4 % (PACKED_DW_WARPS * 32) == 0,
    "packed obs: dW grad pairs split evenly over threads");
static_assert(sizeof(precision_t) == 2, "packed obs: bf16 builds only");
static_assert(OBS_SIZE % 2 == 0 && OBS_PACKED_SCALARS % 2 == 0
    && OBS_PACKED_FEATURES % 2 == 0, "packed obs: features come in bf16 pairs");
static_assert(OBS_PACKED_FEATURES <= PACKED_K, "packed obs: at most 128 features");
static_assert(OBS_PACKED_FEATURES <= OBS_PACKED_SCALARS + 32 * OBS_PACKED_WORDS,
    "packed obs: features past the packed words");

struct PackedEncoderActivations {
    EncoderActivations base;
    Float partials;  // (PACKED_DW_BLOCKS, H, PACKED_K) dW per row chunk
};

// Feature k of a staged packed row as bf16 bits: scalars raw, bits as 1.0 (0x3F80).
__device__ __forceinline__ unsigned int packed_feature(const unsigned int* row, int k) {
    if (k < OBS_PACKED_SCALARS) {
        return (row[k / 2] >> (16 * (k % 2))) & 0xFFFFu;
    }
    if (k >= OBS_PACKED_FEATURES) {
        return 0u;
    }
    int b = k - OBS_PACKED_SCALARS;
    return ((row[OBS_PACKED_SCALARS / 2 + b / 32] >> (b % 32)) & 1u) * 0x3F80u;
}

// Features (k, k + 1) = (k_lo + 2t, + 1) of a staged row as bf16x2. k_lo (a multiple
// of 8) is a compile-time constant in the unrolled callers, so only the groups that
// straddle the scalar/bit or bit/padding boundary take the general path.
__device__ __forceinline__ unsigned int packed_pair(const unsigned int* row, int k_lo, int t) {
    int k = k_lo + 2 * t;
    bool scalars = k_lo + 8 <= OBS_PACKED_SCALARS;
    bool bits = k_lo >= OBS_PACKED_SCALARS && k_lo + 8 <= OBS_PACKED_FEATURES;
    if (scalars || (!bits && k < OBS_PACKED_SCALARS)) {
        return row[k / 2];
    }
    if (!bits && k >= OBS_PACKED_FEATURES) {
        return 0u;
    }
    unsigned int b = k - OBS_PACKED_SCALARS;
    unsigned int w = row[OBS_PACKED_SCALARS / 2 + b / 32] >> (b % 32);
    return (w & 1u) * 0x3F80u | ((w >> 1) & 1u) * 0x3F800000u;
}

// Lane / thread `tid` of `n` loads its words tid + n * u of packed rows
// [r0, r0 + rows) (zero past R) into v, all loads in flight before any use;
// packed_store then writes them to shared memory. rows and n are compile-time.
__device__ __forceinline__ void packed_load(unsigned int* v,
        const precision_t* __restrict__ obs, long r0, long R, int rows, int tid, int n) {
    const unsigned int* src = (const unsigned int*)(obs + r0 * OBS_SIZE);
    long valid = (R - r0) * PACKED_ROW_WORDS;
    #pragma unroll
    for (int u = 0; u < (rows * PACKED_ROW_WORDS + n - 1) / n; u++) {
        int i = tid + n * u;
        v[u] = i < rows * PACKED_ROW_WORDS && i < valid ? src[i] : 0u;
    }
}

__device__ __forceinline__ void packed_store(unsigned int* P, const unsigned int* v,
        int rows, int tid, int n) {
    #pragma unroll
    for (int u = 0; u < (rows * PACKED_ROW_WORDS + n - 1) / n; u++) {
        if (tid + n * u < rows * PACKED_ROW_WORDS) {
            P[tid + n * u] = v[u];
        }
    }
}

// dW pass inputs at r0 into registers: packed rows (PACKED_DW_ROWS) and, per
// thread, PACKED_DW_GRAD (h, h + 1) words of grad for two consecutive rows.
__device__ __forceinline__ void packed_dw_load(unsigned int* pv, unsigned int* gv,
        const precision_t* __restrict__ obs, const unsigned int* __restrict__ gp,
        long r0, long hi, int h_lo, int H) {
    packed_load(pv, obs, r0, hi, PACKED_DW_ROWS, threadIdx.x, PACKED_DW_WARPS * 32);
    #pragma unroll
    for (int u = 0; u < PACKED_DW_GRAD; u++) {
        int i = threadIdx.x + PACKED_DW_WARPS * 32 * u;
        int h = h_lo + 2 * (i % (PACKED_N / 2));
        long r = r0 + 2 * (i / (PACKED_N / 2));
        gv[2 * u] = h < H && r < hi ? gp[(r * H + h) / 2] : 0u;
        gv[2 * u + 1] = h < H && r + 1 < hi ? gp[((r + 1) * H + h) / 2] : 0u;
    }
}

// out (R, H) = unpack(obs) @ W^T for outputs [32y, 32y + 32): the W slice is staged
// once per block in mma B-fragment order; warps walk 16-row tiles, loading the next
// tile into registers while the current one computes.
__global__ void __launch_bounds__(PACKED_WARPS * 32) packed_encoder_fwd(
        precision_t* __restrict__ out, const precision_t* __restrict__ obs,
        const precision_t* __restrict__ W, long R, int H) {
    __shared__ unsigned int Wf[PACKED_K / 16 * PACKED_N / 8 * 64];
    __shared__ unsigned int P[PACKED_WARPS][16 * PACKED_ROW_WORDS];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    int n_lo = blockIdx.y * PACKED_N;
    unsigned int* Pw = P[warp];
    long stride = 16L * gridDim.x * PACKED_WARPS;
    long r0 = 16L * (blockIdx.x * PACKED_WARPS + warp);
    unsigned int next[PACKED_FWD_WORDS];
    packed_load(next, obs, r0, R, 16, lane, 32);
    const unsigned int* Wp = (const unsigned int*)W;
    // Wf[((kk * PACKED_N / 8 + j) * 2 + q) * 32 + l]: register b_q of lane l for
    // k-step kk, n-tile j.
    #pragma unroll
    for (int u = 0; u < PACKED_K / 16 * PACKED_N / 8 * 64 / (PACKED_WARPS * 32); u++) {
        int i = threadIdx.x + PACKED_WARPS * 32 * u;
        int l = i % 32, q = i / 32 % 2, j = i / 64 % (PACKED_N / 8);
        int kk = i / (64 * (PACKED_N / 8));
        int h = n_lo + 8 * j + l / 4, k = 16 * kk + 8 * q + 2 * (l % 4);
        Wf[i] = h < H && k < OBS_PACKED_FEATURES
            ? Wp[((long)h * OBS_PACKED_FEATURES + k) / 2] : 0u;
    }
    packed_store(Pw, next, 16, lane, 32);
    __syncthreads();
    const unsigned int* row0 = Pw + g * PACKED_ROW_WORDS;
    const unsigned int* row1 = row0 + 8 * PACKED_ROW_WORDS;
    for (; r0 < R; r0 += stride) {
        packed_load(next, obs, r0 + stride, R, 16, lane, 32);
        float acc[PACKED_N / 8][4] = {};
        #pragma unroll
        for (int kk = 0; kk < PACKED_K / 16; kk++) {
            unsigned int a[4] = {
                packed_pair(row0, 16 * kk, t), packed_pair(row1, 16 * kk, t),
                packed_pair(row0, 16 * kk + 8, t), packed_pair(row1, 16 * kk + 8, t),
            };
            #pragma unroll
            for (int j = 0; j < PACKED_N / 8; j++) {
                const unsigned int* b = Wf + (kk * (PACKED_N / 8) + j) * 64 + lane;
                mma_16816(acc[j], a, b[0], b[32]);
            }
        }
        #pragma unroll
        for (int j = 0; j < PACKED_N / 8; j++) {
            int c = n_lo + 8 * j + 2 * t;
            if (c < H && r0 + g < R) {
                *(__nv_bfloat162*)(out + (r0 + g) * H + c) =
                    __floats2bfloat162_rn(acc[j][0], acc[j][1]);
            }
            if (c < H && r0 + g + 8 < R) {
                *(__nv_bfloat162*)(out + (r0 + g + 8) * H + c) =
                    __floats2bfloat162_rn(acc[j][2], acc[j][3]);
            }
        }
        __syncwarp();
        packed_store(Pw, next, 16, lane, 32);
        __syncwarp();
    }
}

// partials[blockIdx.x] (H x PACKED_K, fp32) = grad^T @ unpack(obs) over this
// block's row chunk, outputs [32y, 32y + 32). Each 64-row pass stages grad and the
// unpacked features feature-major as row pairs (Gt[h][r / 2], Xt[f][r / 2]), so each
// fragment register is one 32-bit load. Warp w owns h rows 16 (w % 2) + [0, 16)
// and features 32 (w / 2) + [0, 32) as 4 tiles of 16x8.
__global__ void __launch_bounds__(PACKED_DW_WARPS * 32) packed_encoder_dw(
        float* __restrict__ partials, const precision_t* __restrict__ grad,
        const precision_t* __restrict__ obs, long R, int H) {
    __shared__ unsigned int P[PACKED_DW_ROWS * PACKED_ROW_WORDS];
    __shared__ unsigned int Gt[PACKED_N * PACKED_TLD];
    __shared__ unsigned int Xt[PACKED_K * PACKED_TLD];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    int h_lo = blockIdx.y * PACKED_N, m = 16 * (warp % 2), f0 = 32 * (warp / 2);
    long chunk = (R + gridDim.x - 1) / gridDim.x;
    chunk = (chunk + PACKED_DW_ROWS - 1) / PACKED_DW_ROWS * PACKED_DW_ROWS;
    long lo = blockIdx.x * chunk, hi = min(R, lo + chunk);
    const unsigned int* gp = (const unsigned int*)grad;
    unsigned int pv[PACKED_DW_WORDS], gv[2 * PACKED_DW_GRAD];
    float acc[4][4] = {};
    packed_dw_load(pv, gv, obs, gp, lo, hi, h_lo, H);
    for (long r0 = lo; r0 < hi; r0 += PACKED_DW_ROWS) {
        __syncthreads();
        packed_store(P, pv, PACKED_DW_ROWS, threadIdx.x, PACKED_DW_WARPS * 32);
        // Gt[h][pr] and Gt[h + 1][pr] from the (h, h + 1) words of rows 2pr, 2pr + 1.
        #pragma unroll
        for (int u = 0; u < PACKED_DW_GRAD; u++) {
            int i = threadIdx.x + PACKED_DW_WARPS * 32 * u;
            int h = 2 * (i % (PACKED_N / 2)), pr = i / (PACKED_N / 2);
            Gt[h * PACKED_TLD + pr] = __byte_perm(gv[2 * u], gv[2 * u + 1], 0x5410);
            Gt[(h + 1) * PACKED_TLD + pr] = __byte_perm(gv[2 * u], gv[2 * u + 1], 0x7632);
        }
        __syncthreads();
        if (r0 + PACKED_DW_ROWS < hi) {
            packed_dw_load(pv, gv, obs, gp, r0 + PACKED_DW_ROWS, hi, h_lo, H);
        }
        // Lane = scalar / bit index within a word; warps split the row pairs.
        for (int pr = warp; pr < PACKED_DW_ROWS / 2; pr += PACKED_DW_WARPS) {
            const unsigned int* ra = P + 2 * pr * PACKED_ROW_WORDS;
            const unsigned int* rb = ra + PACKED_ROW_WORDS;
            if (lane < OBS_PACKED_SCALARS) {
                Xt[lane * PACKED_TLD + pr] =
                    packed_feature(ra, lane) | packed_feature(rb, lane) << 16;
            }
            #pragma unroll
            for (int w = 0; w < OBS_PACKED_WORDS; w++) {
                int f = OBS_PACKED_SCALARS + 32 * w + lane;
                if (f < PACKED_K) {
                    unsigned int ba = ra[OBS_PACKED_SCALARS / 2 + w] >> lane & 1u;
                    unsigned int bb = rb[OBS_PACKED_SCALARS / 2 + w] >> lane & 1u;
                    Xt[f * PACKED_TLD + pr] = f < OBS_PACKED_FEATURES
                        ? ba * 0x3F80u | bb * 0x3F800000u : 0u;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kr = 0; kr < PACKED_DW_ROWS; kr += 16) {
            // Rows kr + 2t (+1) are pair kr / 2 + t; rows kr + 2t + 8 (+1) are 4 later.
            int pa = kr / 2 + t;
            const unsigned int* ga = Gt + (m + g) * PACKED_TLD + pa;
            unsigned int a[4] = {
                ga[0], ga[8 * PACKED_TLD], ga[4], ga[8 * PACKED_TLD + 4],
            };
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const unsigned int* xb = Xt + (f0 + 8 * j + g) * PACKED_TLD + pa;
                mma_16816(acc[j], a, xb[0], xb[4]);
            }
        }
    }
    if (h_lo + m >= H) {
        return;
    }
    float* p = partials + ((long)blockIdx.x * H + h_lo + m) * PACKED_K;
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        int f = f0 + 8 * j + 2 * t;
        *(float2*)(p + g * PACKED_K + f) = make_float2(acc[j][0], acc[j][1]);
        *(float2*)(p + (g + 8) * PACKED_K + f) = make_float2(acc[j][2], acc[j][3]);
    }
}

// wgrad (H, OBS_PACKED_FEATURES) = row-chunk partials summed in a fixed order.
__global__ void packed_encoder_dw_reduce(precision_t* __restrict__ wgrad,
        const float* __restrict__ partials, int H) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H * OBS_PACKED_FEATURES) {
        return;
    }
    int h = i / OBS_PACKED_FEATURES, k = i % OBS_PACKED_FEATURES;
    float s = 0.0f;
    for (int b = 0; b < PACKED_DW_BLOCKS; b++) {
        s += partials[((long)b * H + h) * PACKED_K + k];
    }
    wgrad[i] = from_float(s);
}

Prec packed_encoder_forward(void* w, void* activations, Prec input, cudaStream_t stream) {
    EncoderWeights* ew = (EncoderWeights*)w;
    EncoderActivations* a = (EncoderActivations*)activations;
    if (a->wgrad_scratch.data) {
        a->saved_input = input;
    }
    long R = numel(input.shape) / OBS_SIZE;
    int H = ew->out_dim;
    // Warps walk 16-row tiles: enough blocks to fill the SMs, few enough that
    // each block's staged W slice serves several tiles.
    static int sms = 0;
    if (sms == 0) {
        int dev;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    }
    long tiles = (R + 15) / 16;
    dim3 grid(min((tiles + PACKED_WARPS - 1) / PACKED_WARPS, 4L * sms),
        (H + PACKED_N - 1) / PACKED_N);
    packed_encoder_fwd<<<grid, PACKED_WARPS * 32, 0, stream>>>(
        a->out.data, input.data, ew->weight.data, R, H);
    return a->out;
}

void packed_encoder_backward(void* w, void* activations, Prec grad, cudaStream_t stream) {
    EncoderWeights* ew = (EncoderWeights*)w;
    PackedEncoderActivations* a = (PackedEncoderActivations*)activations;
    long R = numel(a->base.saved_input.shape) / OBS_SIZE;
    int H = ew->out_dim;
    dim3 grid(PACKED_DW_BLOCKS, (H + PACKED_N - 1) / PACKED_N);
    packed_encoder_dw<<<grid, PACKED_DW_WARPS * 32, 0, stream>>>(
        a->partials.data, grad.data, a->base.saved_input.data, R, H);
    packed_encoder_dw_reduce<<<grid_size(H * OBS_PACKED_FEATURES), BLOCK_SIZE, 0, stream>>>(
        a->base.wgrad_scratch.data, a->partials.data, H);
}

void packed_encoder_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    EncoderWeights* ew = (EncoderWeights*)w;
    PackedEncoderActivations* a = (PackedEncoderActivations*)activations;
    encoder_reg_train(w, &a->base, acts, grads, B_TT);
    a->partials = {.shape = {PACKED_DW_BLOCKS, ew->out_dim, PACKED_K}};
    alloc_register(acts, &a->partials);
}
#endif

#if !defined(PRECISION_FLOAT)
// Decoders with at most DEC_SMALL_OUT outputs (H % 8 == 0). Backward: dX straight
// from the fp32 grads beats assemble + cuBLAS (K = od + 1) about 2x at every H.
// Forward up to H = DEC_SMALL_FWD_H: a direct kernel ties cuBLAS in isolation at
// H = 32 but beats it next to the concurrent learner / rollout (6 vs 10 us per
// breakout rollout step); cuBLAS wins clearly from H = 128 on.
#define DEC_SMALL_OUT 16
#define DEC_SMALL_FWD_H 64

__device__ __forceinline__ void dec_ld8(const precision_t* p, float* o) {
    uint4 v = *(const uint4*)p;
    const __nv_bfloat162* h = (const __nv_bfloat162*)&v;
    #pragma unroll
    for (int q = 0; q < 4; q++) {
        float2 f = __bfloat1622float2(h[q]);
        o[2 * q] = f.x;
        o[2 * q + 1] = f.y;
    }
}

// out (R, od1) = X (R, H) @ W^T: thread = row, one pass over its X row; W staged in
// shared memory (warp-uniform broadcast reads).
__global__ void decoder_fwd_small(precision_t* __restrict__ out,
        const precision_t* __restrict__ X, const precision_t* __restrict__ W,
        int R, int H, int od1) {
    extern __shared__ __align__(16) precision_t dec_w[];
    for (int i = threadIdx.x; i < od1 * H / 8; i += blockDim.x) {
        ((uint4*)dec_w)[i] = ((const uint4*)W)[i];
    }
    __syncthreads();
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= R) {
        return;
    }
    const precision_t* x = X + (long)r * H;
    float acc[DEC_SMALL_OUT] = {};
    for (int c = 0; c < H; c += 8) {
        float xv[8];
        dec_ld8(x + c, xv);
        #pragma unroll
        for (int j = 0; j < DEC_SMALL_OUT; j++) {
            if (j < od1) {
                float wv[8];
                dec_ld8(dec_w + j * H + c, wv);
                #pragma unroll
                for (int q = 0; q < 8; q++) {
                    acc[j] += xv[q] * wv[q];
                }
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < DEC_SMALL_OUT; j++) {
        if (j < od1) {
            out[(long)r * od1 + j] = from_float(acc[j]);
        }
    }
}

// dX (R, H) = G @ W for G = [grad_logits (R, od) | grad_value (R)], from the fp32
// grads: thread = (row, 8 columns). Column chunk 0 also stores G as bf16 (g_out) for
// the dW GEMM.
__global__ void decoder_dx_small(precision_t* __restrict__ dx,
        precision_t* __restrict__ g_out, const float* __restrict__ grad_logits,
        const float* __restrict__ grad_value, const precision_t* __restrict__ W,
        int R, int H, int od) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R * (H / 8)) {
        return;
    }
    long r = i / (H / 8);
    int c = 8 * (i % (H / 8));
    float acc[8] = {};
    for (int j = 0; j <= od; j++) {
        float gj = j < od ? grad_logits[r * od + j] : grad_value[r];
        if (c == 0) {
            g_out[r * (od + 1) + j] = from_float(gj);
        }
        float wv[8];
        dec_ld8(W + (long)j * H + c, wv);
        #pragma unroll
        for (int q = 0; q < 8; q++) {
            acc[q] += gj * wv[q];
        }
    }
    uint4 v;
    __nv_bfloat162* h = (__nv_bfloat162*)&v;
    #pragma unroll
    for (int q = 0; q < 4; q++) {
        h[q] = __floats2bfloat162_rn(acc[2 * q], acc[2 * q + 1]);
    }
    *(uint4*)(dx + r * H + c) = v;
}
#endif

Prec decoder_forward(void* w, void* activations, Prec input, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    if (a->wgrad_scratch.data) {
        a->saved_input = input;
    }
#if !defined(PRECISION_FLOAT)
    int H = dw->hidden_dim, od1 = dw->output_dim + 1;
    if (od1 <= DEC_SMALL_OUT && H % 8 == 0 && H <= DEC_SMALL_FWD_H) {
        int R = (int)(numel(input.shape) / H);
        decoder_fwd_small<<<grid_size(R), BLOCK_SIZE, od1 * H * sizeof(precision_t),
            stream>>>(a->out.data, input.data, dw->weight.data, R, H, od1);
        return a->out;
    }
#endif
    puf_mm(&input, &dw->weight, &a->out, stream);
    return a->out;
}

void decoder_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    Prec wt = {
        .data = dw->weight.data,
        .shape = {dw->output_dim + 1, dw->hidden_dim},
    };
    puf_kaiming_init(&wt, 1.0f, (*seed)++, stream);
}

void decoder_reg_params(void* w, Allocator* alloc) {
    DecoderWeights* dw = (DecoderWeights*)w;
    dw->weight = {.shape = {dw->output_dim + 1, dw->hidden_dim}};
    alloc_register(alloc, &dw->weight);
    if (dw->continuous) {
        dw->logstd = {.shape = {1, dw->output_dim}};
        alloc_register(alloc,&dw->logstd);
    }
}

void decoder_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    int od1 = dw->output_dim + 1;
    *a = (DecoderActivations){
        .out =              {.shape = {B_TT, od1}},
        .grad_out =         {.shape = {B_TT, od1}},
        .grad_input =       {.shape = {B_TT, dw->hidden_dim}},
        .wgrad_scratch =    {.shape = {od1, dw->hidden_dim}},
        .logstd_scratch =   {.shape = {1, dw->output_dim}},
    };
    alloc_register(acts, &a->out);
    alloc_register(acts, &a->grad_out);
    alloc_register(acts, &a->grad_input);
    alloc_register(grads, &a->wgrad_scratch);
    if (dw->continuous) {
        alloc_register(grads, &a->logstd_scratch);
    }
}

void decoder_reg_rollout(void* w, void* activations, Allocator* alloc, int B) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    a->out = {.shape = {B, dw->output_dim + 1}};
    alloc_register(alloc, &a->out);
}

void* decoder_create_weights(void* self) {
    Decoder* d = (Decoder*)self;
    DecoderWeights* dw = (DecoderWeights*)calloc(1, sizeof(DecoderWeights));
    dw->hidden_dim = d->hidden_dim;
    dw->output_dim = d->output_dim;
    dw->continuous = d->continuous;
    return dw;
}

Prec decoder_backward(void* w, void* activations, Float grad_logits,
        Float grad_logstd, Float grad_value, cudaStream_t stream) {
    DecoderWeights* dw = (DecoderWeights*)w;
    DecoderActivations* a = (DecoderActivations*)activations;
    int B_TT = a->saved_input.shape[0];
    int od = dw->output_dim, od1 = od + 1, H = dw->hidden_dim;
    // grad_out = G = [grad_logits | grad_value] in bf16, the dW GEMM's input. Small
    // decoders compute dX from the fp32 grads in the same pass.
    bool small = false;
#if !defined(PRECISION_FLOAT)
    small = od1 <= DEC_SMALL_OUT && H % 8 == 0;
    if (small) {
        decoder_dx_small<<<grid_size(B_TT * (H / 8)), BLOCK_SIZE, 0, stream>>>(
            a->grad_input.data, a->grad_out.data, grad_logits.data, grad_value.data,
            dw->weight.data, B_TT, H, od);
    }
#endif
    if (!small) {
        assemble_decoder_grad<<<grid_size(B_TT * od1), BLOCK_SIZE, 0, stream>>>(
            a->grad_out.data, grad_logits.data, grad_value.data, B_TT, od, od1);
    }
    // dW on the side stream; dX on main (needed for the residual chain).
    puf_mm_tn_async_after(&a->grad_out, &a->saved_input, &a->wgrad_scratch, stream);
    if (dw->continuous && grad_logstd.data != NULL) {
        sum_rows_to_precision_kernel<<<dw->output_dim, 256, 0, stream>>>(
            a->logstd_scratch.data, grad_logstd.data, B_TT, dw->output_dim);
    }
    if (!small) {
        puf_mm_nn(&a->grad_out, &dw->weight, &a->grad_input, stream);
    }
    return a->grad_input;
}

struct MinGRUActivations {
    // Rollout
    Prec* combined;       // (B rollout, 3*T)[num_layers]
    Prec out;             // (B rollout, T)
    // Training
    Prec* saved_inputs;   // views of each layer's input (B, TT, T)[num_layers]
    PrefixScan* scan_bufs;           // [num_layers]
    Prec* combined_bufs;  // (B*TT, 3*T)[num_layers]
    Prec* wgrad_scratch;  // (3*T, T)[num_layers]
    Prec grad_bufs[2];    // (B*TT, T) ping-pong layer dX (highway + GEMM); fused: [0]
    Prec grad_next_state; // (B, 1, T)
    // Fused path (mingru_fused_*): per-layer per-block dW partials
    // (num_layers, fused_blocks, 3*T*T) and W^T (T, 3*T)[num_layers]
    // written by the forward for the backward's dX; combined buffers stay unallocated.
    bool fused;
    bool fused_all_layers;  // one bwd launch for all layers (partials fit in L2)
    int fused_blocks;
    Float dw_partials;
    Prec* wt;
};

struct MinGRUWeights {
    int hidden, num_layers, horizon;
    Prec* weights;  // [num_layers]
};

Prec mingru_state_layer(Prec& state, int layer, int agent_off, int B) {
    long A = state.shape[1], H = state.shape[2];
    return {
        .data = state.data + ((long)layer * A + agent_off) * H,
        .shape = {B, H},
    };
}

void mingru_init_weights(void* w, ulong* seed, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    for (int i = 0; i < m->num_layers; i++) {
        Prec w2d = {
            .data = m->weights[i].data,
            .shape = {3 * m->hidden, m->hidden},
        };
        puf_kaiming_init(&w2d, 1.0f, (*seed)++, stream);
    }
}

void mingru_reg_params(void* w, Allocator* alloc) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    for (int i = 0; i < m->num_layers; i++) {
        m->weights[i] = {.shape = {3 * m->hidden, m->hidden}};
        alloc_register(alloc, &m->weights[i]);
    }
}

#if !defined(PRECISION_FLOAT)
// The fused H == 32 MinGRU kernels need bf16 tensor cores (sm_80+).
static bool mgf_supported(int H) {
    int dev, major;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    return H == MGF_H && major >= 8;
}
#endif

void mingru_reg_train(void* w, void* activations,
        Allocator* acts, Allocator* grads, int B_TT) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int H = m->hidden, TT = m->horizon, B = B_TT / TT;
    a->saved_inputs = (Prec*)calloc(m->num_layers, sizeof(Prec));
    a->scan_bufs = (PrefixScan*)calloc(m->num_layers, sizeof(PrefixScan));
    a->combined_bufs = (Prec*)calloc(m->num_layers, sizeof(Prec));
    a->wgrad_scratch = (Prec*)calloc(m->num_layers, sizeof(Prec));
#if !defined(PRECISION_FLOAT)
    a->fused = mgf_supported(H) && TT <= MGF_ROWS && MGF_ROWS % TT == 0
        && TT % MGF_U == 0 && B_TT % MGF_ROWS == 0 && m->num_layers <= MGF_MAX_LAYERS;
    if (a->fused) {
        a->fused_blocks = B_TT / MGF_ROWS;
        // One backward launch for all layers keeps each block's rows in shared memory
        // between layers, but its dW partials are summed only afterwards: worth it while
        // they all stay in L2 (3080 Ti, 4 layers at 16K rows: 12 MB of 6 MB, 8% slower
        // end to end). Otherwise one launch per layer, each layer's partials summed on
        // the side stream while still in L2, under the next layer's backward.
        a->dw_partials = {.shape = {m->num_layers, a->fused_blocks, 3 * H * H}};
        int dev, l2_bytes;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, dev);
        a->fused_all_layers =
            numel(a->dw_partials.shape) * (long)sizeof(float) <= l2_bytes / 2;
        a->grad_bufs[0] = {.shape = {B_TT, H}};
        a->grad_bufs[1] = {.shape = {B_TT, H}};
        alloc_register(acts, &a->grad_bufs[0]);
        if (!a->fused_all_layers) {
            alloc_register(acts, &a->grad_bufs[1]);
        }
        alloc_register(acts, &a->dw_partials);
        a->wt = (Prec*)calloc(m->num_layers, sizeof(Prec));
        for (int i = 0; i < m->num_layers; i++) {
            a->scan_bufs[i] = {.B = B, .T = TT, .H = H, .scan_h = {.shape = {B, TT, H}},
                .out = {.shape = {B, TT, H}}};
            a->wgrad_scratch[i] = {.shape = {3 * H, H}};
            a->wt[i] = {.shape = {H, 3 * H}};
            alloc_register(acts, &a->scan_bufs[i].out);
            alloc_register(acts, &a->scan_bufs[i].scan_h);
            alloc_register(acts, &a->wt[i]);
            alloc_register(grads, &a->wgrad_scratch[i]);
        }
        // Four blocks per SM need the full shared-memory carveout.
        cudaFuncSetAttribute(mingru_fused_fwd,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        cudaFuncSetAttribute(mingru_fused_bwd,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
        return;
    }
#endif
    a->grad_bufs[0] = {.shape = {B_TT, H}};
    a->grad_bufs[1] = {.shape = {B_TT, H}};
    a->grad_next_state = {.shape = {B, 1, H}};
    alloc_register(acts, &a->grad_bufs[0]);
    alloc_register(acts, &a->grad_bufs[1]);
    alloc_register(acts, &a->grad_next_state);
    for (int i = 0; i < m->num_layers; i++) {
        a->scan_bufs[i] = {
            .B = B, .T = TT, .H = H,
            .scan_h =           {.shape = {B, TT, H}},
            .out =              {.shape = {B, TT, H}},
            .next_state =       {.shape = {B, 1, H}},
            .grad_combined =    {.shape = {B, TT, 3 * H}},
            .grad_state =       {.shape = {B, 1, H}},
        };
        a->combined_bufs[i] = {.shape = {B_TT, 3 * H}};
        a->wgrad_scratch[i] = {.shape = {3 * H, H}};
        alloc_register(acts, &a->combined_bufs[i]);
        alloc_register(acts, &a->scan_bufs[i].out);
        alloc_register(acts, &a->scan_bufs[i].next_state);
        alloc_register(acts, &a->scan_bufs[i].scan_h);
        alloc_register(acts, &a->scan_bufs[i].grad_combined);
        alloc_register(acts, &a->scan_bufs[i].grad_state);
        alloc_register(grads, &a->wgrad_scratch[i]);
    }
}

void mingru_reg_rollout(void* weights, void* activations,
        Allocator* alloc, int B_inf) {
    MinGRUWeights* w = (MinGRUWeights*)weights;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int H = w->hidden;
    a->out = {.shape = {B_inf, H}};
    alloc_register(alloc, &a->out);
#if !defined(PRECISION_FLOAT)
    a->fused = mgf_supported(H) && w->num_layers <= MGF_MAX_LAYERS;
    if (a->fused) {
        return;
    }
#endif
    a->combined = (Prec*)calloc(w->num_layers, sizeof(Prec));
    for (int i = 0; i < w->num_layers; i++) {
        a->combined[i] = {.shape = {B_inf, 3 * H}};
        alloc_register(alloc, &a->combined[i]);
    }
}

void* mingru_create_weights(void* self) {
    Network* n = (Network*)self;
    MinGRUWeights* mw = (MinGRUWeights*)calloc(1, sizeof(MinGRUWeights));
    mw->hidden = n->hidden;
    mw->num_layers = n->num_layers;
    mw->horizon = n->horizon;
    mw->weights = (Prec*)calloc(n->num_layers, sizeof(Prec));
    return mw;
}

Prec mingru_forward(void* w, Prec x, Prec state,
        void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int B = state.shape[1];
    int H = state.shape[2];
#if !defined(PRECISION_FLOAT)
    if (a->fused) {
        MgfLayers lw;
        for (int i = 0; i < m->num_layers; i++) {
            lw.W[i] = m->weights[i].data;
        }
        mingru_fused_step<<<(B + MGF_ROWS - 1) / MGF_ROWS, MGF_WARPS * 32, MGF_STEP_SMEM,
            stream>>>(a->out.data, state.data, x.data, lw, m->num_layers, B);
        return a->out;
    }
#endif
    for (int i = 0; i < m->num_layers; i++) {
        Prec state_i = mingru_state_layer(state, i, 0, B);
        puf_mm(&x, &m->weights[i], &a->combined[i], stream);
        // In place: each thread reads state/x before writing next state/out.
        mingru_gate<<<grid_size(B*H), BLOCK_SIZE, 0, stream>>>(
            a->out.data, state_i.data,
            a->combined[i].data, state_i.data, x.data, H, B);
        x = a->out;
    }
    return x;
}

Prec mingru_forward_train(void* w, Prec x, Prec state, Prec terminals,
        void* activations, int agent_off, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
    int B = (int)x.shape[0];
#if !defined(PRECISION_FLOAT)
    if (a->fused) {
        MgfFwdLayers p;
        const precision_t* x0 = x.data;
        for (int i = 0; i < m->num_layers; i++) {
            PrefixScan& scan = a->scan_bufs[i];
            a->saved_inputs[i] = x;
            scan.state_ptr = mingru_state_layer(state, i, agent_off, B).data;
            scan.input_ptr = x.data;
            scan.terminals_ptr = terminals.data;
            p.W[i] = m->weights[i].data;
            p.state[i] = scan.state_ptr;
            p.wt[i] = a->wt[i].data;
            p.out[i] = scan.out.data;
            p.hs[i] = scan.scan_h.data;
            x = scan.out;
        }
        mingru_fused_fwd<<<a->fused_blocks, MGF_WARPS * 32, MGF_FWD_SMEM, stream>>>(
            p, m->num_layers, x0, terminals.data, m->horizon);
        return x;
    }
#endif
    for (int i = 0; i < m->num_layers; i++) {
        // Layer inputs (encoder out / prior scan out) persist through backward.
        a->saved_inputs[i] = x;
        Prec state_i = mingru_state_layer(state, i, agent_off, B);
        puf_mm(&x, &m->weights[i], &a->combined_bufs[i], stream);
        PrefixScan& scan = a->scan_bufs[i];
        scan.combined_ptr = a->combined_bufs[i].data;
        scan.state_ptr = state_i.data;
        scan.input_ptr = a->saved_inputs[i].data;
        scan.terminals_ptr = terminals.data;
        if (scan.T >= 256 && scan.B * scan.H <= 32768
                || scan.T >= 128 && scan.H <= 32) {
            dim3 block(MINGRU_TILE, MINGRU_TILE);
            dim3 grid(scan.B, (scan.H + MINGRU_TILE - 1) / MINGRU_TILE);
            mingru_scan_forward_row<<<grid, block, 0, stream>>>(scan);
        } else if (scan.B * scan.H <= 32768 && scan.T % MINGRU_UNROLL == 0) {
            mingru_scan_forward_unroll<<<grid_size(scan.B * scan.H), BLOCK_SIZE, 0,
                stream>>>(scan);
        } else {
            mingru_scan_forward<<<grid_size(scan.B * scan.H), BLOCK_SIZE, 0, stream>>>(scan);
        }
        x = scan.out;
    }
    return x;
}

Prec mingru_backward(void* w, Prec grad, void* activations, cudaStream_t stream) {
    MinGRUWeights* m = (MinGRUWeights*)w;
    MinGRUActivations* a = (MinGRUActivations*)activations;
#if !defined(PRECISION_FLOAT)
    if (a->fused && a->fused_all_layers) {
        MgfBwdLayers p;
        MgfWgrad wgrad;
        for (int i = 0; i < m->num_layers; i++) {
            p.x[i] = a->scan_bufs[i].input_ptr;
            p.W[i] = m->weights[i].data;
            p.Wt[i] = a->wt[i].data;
            p.hs[i] = a->scan_bufs[i].scan_h.data;
            wgrad.p[i] = a->wgrad_scratch[i].data;
        }
        mingru_fused_bwd<<<a->fused_blocks, MGF_WARPS * 32, MGF_BWD_SMEM, stream>>>(
            p, m->num_layers, a->grad_bufs[0].data, a->dw_partials.data,
            a->scan_bufs[0].terminals_ptr, grad.data, m->horizon);
        mingru_fused_dw_reduce<<<dim3(3 * MGF_H * MGF_H / 32, m->num_layers),
            MGF_RED_WARPS * 32, 0, puf_fork(stream)>>>(wgrad, a->dw_partials.data,
            a->fused_blocks);
        return a->grad_bufs[0];
    }
    if (a->fused) {
        long per_layer = (long)a->fused_blocks * 3 * MGF_H * MGF_H;
        for (int i = m->num_layers - 1, k = 0; i >= 0; i--, k ^= 1) {
            MgfBwdLayers p = {};
            p.x[0] = a->scan_bufs[i].input_ptr;
            p.W[0] = m->weights[i].data;
            p.Wt[0] = a->wt[i].data;
            p.hs[0] = a->scan_bufs[i].scan_h.data;
            float* part = a->dw_partials.data + i * per_layer;
            mingru_fused_bwd<<<a->fused_blocks, MGF_WARPS * 32, MGF_BWD_SMEM, stream>>>(
                p, 1, a->grad_bufs[k].data, part, a->scan_bufs[i].terminals_ptr, grad.data,
                m->horizon);
            MgfWgrad wgrad = {};
            wgrad.p[0] = a->wgrad_scratch[i].data;
            mingru_fused_dw_reduce<<<3 * MGF_H * MGF_H / 32, MGF_RED_WARPS * 32, 0,
                puf_fork(stream)>>>(wgrad, part, a->fused_blocks);
            grad = a->grad_bufs[k];
        }
        return grad;
    }
#endif
    for (int i = m->num_layers - 1, k = 0; i >= 0; i--, k ^= 1) {
        // Scan writes the highway grad into this layer's dX buffer and the dX
        // GEMM accumulates onto it (beta=1). Ping-pong: never write the grad read.
        PrefixScan& scan = a->scan_bufs[i];
        scan.grad_input = a->grad_bufs[k];
        if (scan.T >= 256 && scan.B * scan.H <= 32768
                || scan.T >= 128 && scan.H <= 32) {
            dim3 block(MINGRU_TILE, MINGRU_TILE);
            dim3 grid(scan.B, (scan.H + MINGRU_TILE - 1) / MINGRU_TILE);
            mingru_scan_backward_row<<<grid, block, 0, stream>>>(
                scan, grad.data, a->grad_next_state.data);
        } else if (scan.B * scan.H <= 32768 && scan.T % MINGRU_UNROLL == 0) {
            mingru_scan_backward_unroll<<<grid_size(scan.B * scan.H), BLOCK_SIZE, 0,
                stream>>>(scan, grad.data, a->grad_next_state.data);
        } else {
            mingru_scan_backward<<<grid_size(scan.B * scan.H), BLOCK_SIZE, 0,
                stream>>>(scan, grad.data, a->grad_next_state.data);
        }
        // dW on side stream (per-layer scratch); dX on main continues the bwd chain.
        puf_mm_tn_async_after(&scan.grad_combined, &a->saved_inputs[i],
            &a->wgrad_scratch[i], stream);
        puf_mm_nn(&scan.grad_combined, &m->weights[i], &a->grad_bufs[k], stream,
            1.0f, 1.0f);
        grad = a->grad_bufs[k];
    }
    return grad;
}

struct Arch {
    Encoder encoder;
    Decoder decoder;
    Network network;
};

struct Activations {
    void* encoder;
    void* decoder;
    void* network;
};

struct Weights {
    void* encoder;
    void* decoder;
    void* network;
    int head_lo, head_hi;  // decoder's param regs [lo, hi) (output heads + logstd)
};

Prec arch_forward(Arch* p, Weights& w, Activations& activations,
        Prec obs, Prec state, cudaStream_t stream) {
    Prec enc_out = p->encoder.forward(
        w.encoder, activations.encoder, obs, stream);
    Prec h = p->network.forward(
        w.network, enc_out, state, activations.network, stream);
    return p->decoder.forward(w.decoder, activations.decoder, h, stream);
}

void arch_backward(Arch* p, Weights& w,
        Activations& activations, Float grad_logits,
        Float grad_logstd, Float grad_value, cudaStream_t stream) {
    int B = grad_logits.shape[0], TT = grad_logits.shape[1];
    Prec grad_h = p->decoder.backward(w.decoder,
        activations.decoder, *puf_squeeze(&grad_logits, 0),
        grad_logstd, *puf_squeeze(&grad_value, 0), stream);
    grad_h = p->network.backward(w.network,
        *puf_unsqueeze(&grad_h, 0, B, TT), activations.network, stream);
    p->encoder.backward(w.encoder, activations.encoder, grad_h, stream);
    puf_dw_join(stream); // sycn dW GEMMs before muon reads weight grads.
}

Activations arch_reg_train(Arch* p, Weights& w,
        Allocator* acts, Allocator* grads, int B_TT) {
    Activations a;
    a.encoder = calloc(1, p->encoder.activation_size);
    a.decoder = calloc(1, p->decoder.activation_size);
    a.network = calloc(1, sizeof(MinGRUActivations));
    p->encoder.reg_train(w.encoder, a.encoder, acts, grads, B_TT);
    p->decoder.reg_train(w.decoder, a.decoder, acts, grads, B_TT);
    p->network.reg_train(w.network, a.network, acts, grads, B_TT);
    return a;
}

Activations arch_reg_rollout(Arch* p, Weights& w,
        Allocator* acts, int B_inf) {
    Activations a;
    a.encoder = calloc(1, p->encoder.activation_size);
    a.decoder = calloc(1, p->decoder.activation_size);
    a.network = calloc(1, sizeof(MinGRUActivations));
    p->encoder.reg_rollout(w.encoder, a.encoder, acts, B_inf);
    p->decoder.reg_rollout(w.decoder, a.decoder, acts, B_inf);
    p->network.reg_rollout(w.network, a.network, acts, B_inf);
    return a;
}

void weights_init(Arch* p, Weights& w,
        uint64_t* seed, cudaStream_t stream) {
    p->encoder.init_weights(w.encoder, seed, stream);
    p->decoder.init_weights(w.decoder, seed, stream);
    p->network.init_weights(w.network, seed, stream);
}

Weights weights_create(Arch* p, Allocator* params) {
    Weights w;
    w.encoder = p->encoder.create_weights(&p->encoder);
    w.decoder = p->decoder.create_weights(&p->decoder);
    w.network = p->network.create_weights(&p->network);
    p->encoder.reg_params(w.encoder, params);
    w.head_lo = params->num_regs;
    p->decoder.reg_params(w.decoder, params);
    w.head_hi = params->num_regs;
    p->network.reg_params(w.network, params);
    return w;
}

// Custom architectures for specific envs. Not yet
// happy with the API, but we can't narrow it yet
// because fast, general encoder arch is an
// unsolved research problem.
#include "ocean.cu"

// Build an Arch (ops + dims) for this env. Encoder/decoder algorithms are
// fixed at compile time; hidden_size/num_layers/horizon parameterize shape.
// Arch has no heap state so this returns by value; callers store it wherever.
Arch build_arch(int input_size, int hidden_size,
        int num_layers, int decoder_output_size, bool is_continuous, int horizon) {
    Encoder encoder = {
        .forward = encoder_forward,
        .backward = encoder_backward,
        .init_weights = encoder_init_weights,
        .reg_params = encoder_reg_params,
        .reg_train = encoder_reg_train,
        .reg_rollout = encoder_reg_rollout,
        .create_weights = encoder_create_weights,
        .in_dim = input_size, .out_dim = hidden_size,
        .activation_size = sizeof(EncoderActivations),
    };
#ifdef OBS_PACKED_FEATURES
    assert(hidden_size % 16 == 0 && "packed obs: hidden_size must be a multiple of 16");
    encoder.forward = packed_encoder_forward;
    encoder.backward = packed_encoder_backward;
    encoder.reg_train = packed_encoder_reg_train;
    encoder.in_dim = OBS_PACKED_FEATURES;
    encoder.activation_size = sizeof(PackedEncoderActivations);
#endif
    create_custom_encoder(&encoder);
    Decoder decoder = {
        .forward = decoder_forward,
        .backward = decoder_backward,
        .init_weights = decoder_init_weights,
        .reg_params = decoder_reg_params,
        .reg_train = decoder_reg_train,
        .reg_rollout = decoder_reg_rollout,
        .create_weights = decoder_create_weights,
        .hidden_dim = hidden_size,
        .output_dim = decoder_output_size,
        .continuous = is_continuous,
        .activation_size = sizeof(DecoderActivations),
    };
    create_custom_decoder(&decoder);
    Network network = {
        .forward = mingru_forward,
        .forward_train = mingru_forward_train,
        .backward = mingru_backward,
        .init_weights = mingru_init_weights,
        .reg_params = mingru_reg_params,
        .reg_train = mingru_reg_train,
        .reg_rollout = mingru_reg_rollout,
        .create_weights = mingru_create_weights,
        .hidden = hidden_size,
        .num_layers = num_layers,
        .horizon = horizon,
    };
    return Arch{
        .encoder = encoder, .decoder = decoder, .network = network,
    };
}

// grid (nparts, count): sum-of-squares partials of matrix blockIdx.y (n elems
// each, contiguous) into partials[blockIdx.y * nparts + blockIdx.x].
__global__ void muon_sum_sq_partials(float* __restrict__ partials,
        const precision_t* __restrict__ src, int n) {
    __shared__ float sdata[256];
    const precision_t* s = src + (long)blockIdx.y * n;
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += blockDim.x * gridDim.x) {
        float v = to_float(s[i]);
        sum += v * v;
    }
    sdata[tid] = sum;
    block_reduce_sum(sdata, &partials[blockIdx.y * gridDim.x + blockIdx.x], tid,
        blockDim.x, 1);
}

// Total of <= 256 partials. Every 256-thread block re-reduces them instead of
// paying a separate reduce launch per norm. All threads must call it.
__device__ float muon_sum_partials(const float* partials, int nparts) {
    __shared__ float sdata[256];
    __shared__ float total;
    int tid = threadIdx.x;
    sdata[tid] = tid < nparts ? partials[tid] : 0.0f;
    block_reduce_sum(sdata, &total, tid, blockDim.x, 1);
    __syncthreads();
    return total;
}

// Adam second-moment decay for head params; their beta1 is the Muon momentum.
#define HEAD_ADAM_B2 0.999f

// Global grad clip by L2, then per-element optimizer state: Nesterov momentum for
// Muon params, bias-corrected Adam for flat range [adam_lo, adam_hi) (decoder
// heads + logstd). The Adam update is pre-scaled by adam_ratio so the shared
// w -= lr * update steps by lr * adam_ratio. adam_t: 1-based step count.
__global__ void muon_clip_nesterov(float* __restrict__ mb,
        precision_t* __restrict__ gc, const float* __restrict__ partials, int nparts,
        float max_norm, float eps, float mu, int n, float* __restrict__ vb,
        long adam_lo, long adam_hi, const int* __restrict__ adam_t, float adam_ratio) {
    float sum_sq = muon_sum_partials(partials, nparts);
    float clip_coef = fminf(max_norm / (sqrtf(sum_sq) + eps), 1.0f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= adam_lo && idx < adam_hi) {
        float g = to_float(gc[idx]) * clip_coef;
        float t = (float)*adam_t;
        float m = mu * mb[idx] + (1.0f - mu) * g;
        float v = HEAD_ADAM_B2 * vb[idx - adam_lo] + (1.0f - HEAD_ADAM_B2) * g * g;
        mb[idx] = m;
        vb[idx - adam_lo] = v;
        float m_hat = m / (1.0f - powf(mu, t));
        float v_hat = v / (1.0f - powf(HEAD_ADAM_B2, t));
        gc[idx] = from_float(adam_ratio * m_hat / (sqrtf(v_hat) + 1e-8f));
        return;
    }
    if (idx < n) {
        float g = to_float(gc[idx]) * clip_coef;
        float m = mu * mb[idx] + g;
        mb[idx] = m;
        gc[idx] = from_float(g + mu * m);
    }
}

// grid (blocks, count): dst = src / max(||src||, eps) per matrix (NS input).
__global__ void muon_l2_normalize(precision_t* __restrict__ dst,
        const precision_t* __restrict__ src, const float* __restrict__ partials,
        int nparts, float eps, int n) {
    float sum_sq = muon_sum_partials(partials + blockIdx.y * nparts, nparts);
    float inv_norm = 1.0f / fmaxf(sqrtf(sum_sq), eps);
    long base = (long)blockIdx.y * n;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[base + idx] = from_float(to_float(src[base + idx]) * inv_norm);
    }
}

// NS polynomial P = c*A@A (already in pa) + b*A + a*I, in place over pa. One
// GEMM X <- P X then needs no beta=a copy of X; n = count * M * M.
__global__ void muon_ns_poly(precision_t* __restrict__ pa,
        const precision_t* __restrict__ a, float ca, float cb, int M, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    int rc = idx % (M * M);
    float diag = (rc / M == rc % M) ? ca : 0.0f;
    pa[idx] = from_float(to_float(pa[idx]) + cb * to_float(a[idx]) + diag);
}

// wb = wb * (1 - lr*wd) - lr * update (update already scaled; one call for all
// params), and refresh the precision_t params from it. Float builds alias
// param onto wb (same value written twice), hence no __restrict__. Advances the
// Adam step count (read by the earlier clip kernel of this step).
__global__ void muon_weight_update(float* wb, precision_t* param,
        const precision_t* __restrict__ update,
        const float* __restrict__ lr_ptr, float wd, int n, int* __restrict__ adam_t) {
    float lr = *lr_ptr;
    float wd_scale = 1.0f - lr * wd;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        *adam_t += 1;
    }
    if (idx < n) {
        float w = wb[idx] * wd_scale - lr * to_float(update[idx]);
        wb[idx] = w;
        param[idx] = from_float(w);
    }
}

constexpr double ns_coeffs[5][3] = {
    {4.0848, -6.8946, 2.9270},
    {3.9505, -6.3029, 2.6377},
    {3.7418, -5.5913, 2.3037},
    {2.8769, -3.1427, 1.2046},
    {2.8366, -3.0525, 1.2012},
};

// NS for small matrices (M = min(R, C) <= MUON_SMALL_M), all in one launch, on tf32
// tensor cores with fp32 accumulation: every NS operand stays tf32-rounded in shared
// memory (10-bit mantissa; reference Muon runs NS in bf16, 7-bit). A matrix whose
// wide orientation X (M x N) fits one block's shared memory is one block, synced with
// __syncthreads only. Otherwise (split) every matrix's columns spread over the blocks
// of a cooperative grid: per step each block writes its partial gram to a global
// slot, the grid syncs, and every block sums its matrix's slots, forms the M x M
// polynomial (redundantly) and updates its own columns. The cuBLAS path costs 22
// launch-bound kernels per group (~40% of a train step at H=32). At M = 64 the
// redundant M^3 polynomial per block loses to cuBLAS (drone H=64 layers: 4% slower).
#define MUON_SMALL_M 32
#define MUON_SMALL_MAX 32
#define MUON_SMALL_SMEM (46 * 1024)  // + 1 KB static: under the 48 KB default
#define MUON_SMALL_THREADS 256  // 8 warps: one 16 x 8 tile each of a 32 x 32 product
#define MUON_SMALL_WMAX 144     // widest slice: X, Y (32 x ld) and A, P fill the smem

struct MuonSmall {
    float coef[5][3];
    long off[MUON_SMALL_MAX];      // matrix offsets in the flat grads
    int R[MUON_SMALL_MAX], C[MUON_SMALL_MAX];
    int blk0[MUON_SMALL_MAX + 1];  // first block of each matrix; [count] = grid
};

// Row stride for w columns: w rounded up to the 16-column double mma step, + 4 so
// fragment loads (8 rows x 4 columns) hit distinct banks.
__host__ __device__ constexpr int muon_small_ld(int w) {
    return (w + 15) / 16 * 16 + 4;
}
static_assert(2 * MUON_SMALL_M * (muon_small_ld(MUON_SMALL_WMAX + 1)
    + muon_small_ld(MUON_SMALL_M)) * 4 > MUON_SMALL_SMEM, "slices that fit are <= WMAX");

// Per block: its columns of X and their update (32 x w), gram and polynomial
// (32 x 32), zero outside the matrix.
static long muon_small_smem(long R, long C, long blocks) {
    long w = (max(R, C) + blocks - 1) / blocks;
    return 2L * MUON_SMALL_M * (muon_small_ld((int)w) + muon_small_ld(MUON_SMALL_M))
        * (long)sizeof(float);
}

// x rounded to the nearest tf32, in an fp32 container: mma operands read it exactly.
__device__ __forceinline__ float tf32_round(float x) {
    unsigned int r;
    asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(r) : "f"(x));
    return __uint_as_float(r);
}

__device__ __forceinline__ void mma_tf32(float* d, const unsigned int* a,
        unsigned int b0, unsigned int b1) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// A fragment of mma_tf32 at p = &M[row g][col t] of a row-major tf32 matrix.
__device__ __forceinline__ void muon_frag(unsigned int* a, const float* p, int ld) {
    a[0] = __float_as_uint(p[0]);
    a[1] = __float_as_uint(p[8 * ld]);
    a[2] = __float_as_uint(p[4]);
    a[3] = __float_as_uint(p[8 * ld + 4]);
}

// Accumulator tile out (lane: rows g, g + 8 at columns 2t, 2t + 1), tf32-rounded.
__device__ __forceinline__ void muon_put(float* out, int ld, const float* d) {
    *(float2*)out = make_float2(tf32_round(d[0]), tf32_round(d[1]));
    *(float2*)(out + 8 * ld) = make_float2(tf32_round(d[2]), tf32_round(d[3]));
}

// out (32 x 32) = X @ X^T over w columns of X: warp = one 16 x 8 tile, two mma chains.
__device__ __forceinline__ void muon_small_gram(float* out, int ldo, const float* X,
        int ldx, int w) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    int r0 = 16 * (warp / 4), c0 = 8 * (warp % 4);
    const float* xa = X + (r0 + g) * ldx + t;
    const float* xb = X + (c0 + g) * ldx + t;
    float acc[2][4] = {};
    for (int k = 0; k < w; k += 16) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            unsigned int a[4];
            muon_frag(a, xa + k + 8 * h, ldx);
            mma_tf32(acc[h], a, __float_as_uint(xb[k + 8 * h]),
                __float_as_uint(xb[k + 8 * h + 4]));
        }
    }
    float d[4];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        d[i] = acc[0][i] + acc[1][i];
    }
    muon_put(out + (r0 + g) * ldo + c0 + 2 * t, ldo, d);
}

// P (32 x 32) = a I + b A + c A @ A for symmetric A (A @ A = A @ A^T).
__device__ __forceinline__ void muon_small_poly(float* P, const float* A, int ld,
        float a, float b, float c) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    int r = 16 * (warp / 4) + g, col = 8 * (warp % 4) + 2 * t;
    const float* pa = A + r * ld + t;
    const float* pb = A + (col - 2 * t + g) * ld + t;
    float acc[2][4] = {};
    #pragma unroll
    for (int k = 0; k < MUON_SMALL_M; k += 16) {
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            unsigned int f[4];
            muon_frag(f, pa + k + 8 * h, ld);
            mma_tf32(acc[h], f, __float_as_uint(pb[k + 8 * h]),
                __float_as_uint(pb[k + 8 * h + 4]));
        }
    }
    const float* ar = A + r * ld + col;
    float d[4] = {
        c * (acc[0][0] + acc[1][0]) + b * ar[0] + (r == col ? a : 0.0f),
        c * (acc[0][1] + acc[1][1]) + b * ar[1] + (r == col + 1 ? a : 0.0f),
        c * (acc[0][2] + acc[1][2]) + b * ar[8 * ld] + (r + 8 == col ? a : 0.0f),
        c * (acc[0][3] + acc[1][3]) + b * ar[8 * ld + 1] + (r + 8 == col + 1 ? a : 0.0f),
    };
    muon_put(P + r * ld + col, ld, d);
}

// Y (32 x w) = P (32 x 32) @ X: warp = 32 x 8 column strips 8 * warp, 8 * warp + 64,
// ...; P's fragments load once, both row tiles share each X fragment.
__device__ __forceinline__ void muon_small_update(float* Y, const float* P, int ldp,
        const float* X, int ldx, int w) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, t = lane % 4;
    unsigned int f[2][MUON_SMALL_M / 8][4];
    #pragma unroll
    for (int h = 0; h < 2; h++) {
        #pragma unroll
        for (int kk = 0; kk < MUON_SMALL_M / 8; kk++) {
            muon_frag(f[h][kk], P + (16 * h + g) * ldp + 8 * kk + t, ldp);
        }
    }
    for (int c0 = 8 * warp; c0 < w; c0 += 8 * MUON_SMALL_THREADS / 32) {
        float acc[2][4] = {};
        #pragma unroll
        for (int kk = 0; kk < MUON_SMALL_M / 8; kk++) {
            unsigned int b0 = __float_as_uint(X[(8 * kk + t) * ldx + c0 + g]);
            unsigned int b1 = __float_as_uint(X[(8 * kk + t + 4) * ldx + c0 + g]);
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                mma_tf32(acc[h], f[h][kk], b0, b1);
            }
        }
        #pragma unroll
        for (int h = 0; h < 2; h++) {
            muon_put(Y + (16 * h + g) * ldx + c0 + 2 * t, ldx, acc[h]);
        }
    }
}

// Moves the block's slice of X (r < M, c < w) in from g (LOAD; returns its sum of
// squares) or scale * X out to g. Wide X = G: row r is w values from g[r * C + c0],
// one per lane; tall X = G^T: column c is the M <= 32 values of g row c0 + c, one per
// lane. Fully unrolled, so a thread's global loads are in flight together.
template <bool LOAD>
__device__ __forceinline__ float muon_small_io(float* X, int ldx,
        precision_t* __restrict__ g, int C, bool tall, int M, int w, int c0, float scale) {
    constexpr int NW = MUON_SMALL_THREADS / 32, TU = (MUON_SMALL_WMAX + NW - 1) / NW;
    constexpr int WR = MUON_SMALL_M / NW, WP = (MUON_SMALL_WMAX + 31) / 32;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float v[TU > WR * WP ? TU : WR * WP];
    float ss = 0.0f;
    if (tall) {
        #pragma unroll
        for (int u = 0; u < TU; u++) {
            int c = warp + NW * u;
            long i = (long)(c0 + c) * M + lane;
            bool ok = c < w && lane < M;
            if (LOAD) {
                v[u] = ok ? to_float(g[i]) : 0.0f;
            } else if (ok) {
                g[i] = from_float(scale * X[lane * ldx + c]);
            }
        }
        if (LOAD) {
            #pragma unroll
            for (int u = 0; u < TU; u++) {
                int c = warp + NW * u;
                if (c < w && lane < M) {
                    X[lane * ldx + c] = v[u];
                }
                ss += v[u] * v[u];
            }
        }
        return ss;
    }
    #pragma unroll
    for (int q = 0; q < WR; q++) {
        #pragma unroll
        for (int p = 0; p < WP; p++) {
            int r = warp + NW * q, c = lane + 32 * p;
            long i = (long)r * C + c0 + c;
            bool ok = r < M && c < w;
            if (LOAD) {
                v[q * WP + p] = ok ? to_float(g[i]) : 0.0f;
            } else if (ok) {
                g[i] = from_float(scale * X[r * ldx + c]);
            }
        }
    }
    if (LOAD) {
        #pragma unroll
        for (int q = 0; q < WR; q++) {
            #pragma unroll
            for (int p = 0; p < WP; p++) {
                int r = warp + NW * q, c = lane + 32 * p;
                if (r < M && c < w) {
                    X[r * ldx + c] = v[q * WP + p];
                }
                ss += v[q * WP + p] * v[q * WP + p];
            }
        }
    }
    return ss;
}

// Block -> (matrix j, column slice c0 .. c0 + w - 1 of its wide orientation X). SPLIT:
// part holds per block 2 parity slots of MUON_SMALL_M^2 partial-gram floats, then one
// norm partial per block; a slot of parity p is rewritten two steps later, after
// every reader passed the grid sync in between. In place in the flat grads: Nesterov
// update in, scaled NS update out.
template <bool SPLIT>
__global__ void __launch_bounds__(MUON_SMALL_THREADS) muon_ns_small(
        precision_t* __restrict__ grads, float* __restrict__ part, MuonSmall tab) {
    extern __shared__ __align__(16) float sm[];
    __shared__ float red[MUON_SMALL_THREADS / 32];
    int j = 0;
    while ((int)blockIdx.x >= tab.blk0[j + 1]) {
        j++;
    }
    int b0 = tab.blk0[j], nb = tab.blk0[j + 1] - b0, s = blockIdx.x - b0;
    precision_t* g = grads + tab.off[j];
    int R = tab.R[j], C = tab.C[j];
    bool tall = R > C;
    int M = tall ? C : R, N = tall ? R : C;
    int c0 = N * s / nb, w = N * (s + 1) / nb - c0;
    int ldx = muon_small_ld(w), ldm = muon_small_ld(MUON_SMALL_M);
    long slot = MUON_SMALL_M * MUON_SMALL_M;
    float* X = sm;
    float* Y = X + MUON_SMALL_M * ldx;
    float* A = Y + MUON_SMALL_M * ldx;
    float* P = A + MUON_SMALL_M * ldm;
    int tid = threadIdx.x;
    for (int i = tid; i < 2 * MUON_SMALL_M * ldx / 4; i += blockDim.x) {
        ((float4*)X)[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);  // X and Y
    }
    __syncthreads();
    float ss = muon_small_io<true>(X, ldx, g, C, tall, M, w, c0, 1.0f);
    for (int o = 16; o > 0; o >>= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, o);
    }
    if (tid % 32 == 0) {
        red[tid / 32] = ss;
    }
    __syncthreads();
    float total = 0.0f;
    for (int q = 0; q < MUON_SMALL_THREADS / 32; q++) {
        total += red[q];
    }
    if (SPLIT) {
        float* norm_part = part + 2 * gridDim.x * slot;
        if (tid == 0) {
            norm_part[blockIdx.x] = total;
        }
        cooperative_groups::this_grid().sync();
        total = 0.0f;
        for (int q = 0; q < nb; q++) {
            total += norm_part[b0 + q];
        }
    }
    float inv_norm = 1.0f / fmaxf(sqrtf(total), 1e-7f);
    for (int r = tid / 32; r < M; r += MUON_SMALL_THREADS / 32) {
        for (int c = tid % 32; c < w; c += 32) {
            X[r * ldx + c] = tf32_round(X[r * ldx + c] * inv_norm);
        }
    }
    __syncthreads();
    for (int it = 0; it < 5; it++) {
        if (SPLIT) {
            float* gp = part + (it & 1) * gridDim.x * slot;
            muon_small_gram(gp + blockIdx.x * slot, MUON_SMALL_M, X, ldx, w);
            cooperative_groups::this_grid().sync();
            for (int i = tid; i < MUON_SMALL_M * MUON_SMALL_M; i += blockDim.x) {
                float a = 0.0f;
                for (int q = 0; q < nb; q++) {
                    a += gp[(b0 + q) * slot + i];
                }
                A[i / MUON_SMALL_M * ldm + i % MUON_SMALL_M] = tf32_round(a);
            }
        } else {
            muon_small_gram(A, ldm, X, ldx, w);
        }
        __syncthreads();
        muon_small_poly(P, A, ldm, tab.coef[it][0], tab.coef[it][1], tab.coef[it][2]);
        __syncthreads();
        muon_small_update(Y, P, ldm, X, ldx, w);
        __syncthreads();
        float* t = X;
        X = Y;
        Y = t;
    }
    muon_small_io<false>(X, ldx, g, C, tall, M, w, c0,
        tall ? sqrtf((float)R / (float)C) : 1.0f);
}

// Consecutive same-shape matrices with no alignment padding between them are
// contiguous in the flat grads: one strided-batched NS per group (MinGRU layers).
struct MuonGroup {
    int reg, count;  // first param reg, number of matrices
    long R, C;
    bool small;      // NS in the muon_ns_small launch
    int blocks;      // muon_ns_small blocks per matrix
};

// Muon optimizer. Our benchmarks show this is a major
// upgrade over Adam (weight decay not needed in RL).
struct Muon {
    double momentum;
    // Scalars / scratch: raw device ptrs. Tensors: allocator.
    float* lr;
    float* norm_partials;  // 256 per matrix of the largest group
    int* adam_t;           // Adam step count (1-based)
    Float mb;              // flat momentum buffer (param-sized)
    Float vb;              // Adam second moments for the head regs
    Prec gram, gram_buf, x_buf;  // per group: count * (M*M, M*M, R*C)
    int small_smem;        // muon_ns_small dynamic smem; 0 = no small groups
    int ns_blocks;         // muon_ns_small grid
    bool ns_split;         // matrices split over a cooperative grid
    float* ns_part;        // split scratch (partial grams, norms)
    Allocator* param_alloc;
    MuonGroup* groups;
    int num_groups;
    int head_lo, head_hi;  // param regs on Adam instead of Muon (empty = none)
    float head_lr_ratio;   // Adam lr = head_lr_ratio * lr
};

// head_lr_ratio > 0 trains param regs [head_lo, head_hi) (decoder: output heads
// + logstd) with Adam. NS fixes each matrix update's size regardless of the
// gradient's, so e.g. a (1, A) logstd moves lr every step whatever ent_coef is.
void muon_init(Muon* m, Allocator* param_alloc, double momentum, Allocator* alloc,
        int head_lo, int head_hi, float head_lr_ratio) {
    m->momentum = momentum;
    m->param_alloc = param_alloc;
    m->head_lr_ratio = head_lr_ratio;
    if (head_lr_ratio > 0.0f) {
        m->head_lo = head_lo;
        m->head_hi = head_hi;
    }
    m->groups = (MuonGroup*)calloc(param_alloc->num_regs, sizeof(MuonGroup));
    long max_gram = 1, max_x = 1, max_count = 1, head_elems = 0;
    for (int i = 0; i < param_alloc->num_regs; i++) {
        AllocEntry& e = param_alloc->regs[i];
        if (i >= m->head_lo && i < m->head_hi) {
            head_elems += numel(e.shape) + 16;  // + bound on alignment padding
            continue;
        }
        if (ndim(e.shape) < 2) {
            continue;
        }
        long R = e.shape[0], C = numel(e.shape) / R, M = min(R, C);
        MuonGroup* g = m->num_groups ? &m->groups[m->num_groups - 1] : NULL;
        if (g && g->reg + g->count == i && g->R == R && g->C == C
                && R * C * (long)sizeof(precision_t) % 16 == 0) {
            g->count++;
        } else {
            g = &m->groups[m->num_groups++];
            *g = (MuonGroup){.reg = i, .count = 1, .R = R, .C = C};
        }
        max_gram = max(max_gram, g->count * M * M);
        max_x = max(max_x, g->count * R * C);
        max_count = max(max_count, (long)g->count);
    }
    // Small groups (tensor-core NS: sm_80+) share one muon_ns_small launch: a block
    // per matrix when every small matrix fits one block, else each matrix's columns
    // split over SMs / matrices blocks of a cooperative grid (grid <= SMs:
    // co-resident).
    int dev, sms, major;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    int n_small = 0;
    bool fits = true;
    for (int gi = 0; gi < m->num_groups; gi++) {
        MuonGroup* g = &m->groups[gi];
        if (major >= 8 && min(g->R, g->C) <= MUON_SMALL_M
                && n_small + g->count <= MUON_SMALL_MAX) {
            g->small = true;
            n_small += g->count;
            fits = fits && muon_small_smem(g->R, g->C, 1) <= MUON_SMALL_SMEM;
        }
    }
    m->ns_split = !fits;
    int per = fits ? 1 : max(1, sms / max(n_small, 1));
    for (int gi = 0; gi < m->num_groups; gi++) {
        MuonGroup* g = &m->groups[gi];
        if (!g->small) {
            continue;
        }
        g->blocks = (int)min((long)per, max(g->R, g->C));
        long smem = muon_small_smem(g->R, g->C, g->blocks);
        if (smem > MUON_SMALL_SMEM) {
            g->small = false;
            continue;
        }
        m->ns_blocks += g->count * g->blocks;
        m->small_smem = max(m->small_smem, (int)smem);
    }
    if (m->ns_split && m->ns_blocks > 0) {
        cudaMalloc((void**)&m->ns_part,
            (2L * MUON_SMALL_M * MUON_SMALL_M + 1) * m->ns_blocks * sizeof(float));
    }
    int t0 = 1;
    cudaMalloc((void**)&m->lr, sizeof(float));
    cudaMalloc((void**)&m->norm_partials, 256 * max_count * sizeof(float));
    cudaMalloc((void**)&m->adam_t, sizeof(int));
    cudaMemcpy(m->adam_t, &t0, sizeof(int), cudaMemcpyHostToDevice);
    // Flat views span the 16B alignment padding between param tensors.
    m->mb = {.shape = {param_alloc->total_bytes / (long)sizeof(precision_t)}};
    m->vb = {.shape = {max(head_elems, 1L)}};
    m->gram =     {.shape = {max_gram}};
    m->gram_buf = {.shape = {max_gram}};
    m->x_buf =    {.shape = {max_x}};
    alloc_register(alloc, &m->mb);
    alloc_register(alloc, &m->gram);
    alloc_register(alloc, &m->gram_buf);
    alloc_register(alloc, &m->x_buf);
    alloc_register(alloc, &m->vb);
}

// Row-major GEMM for NS over a group: C_i = alpha * op_a(A_i) @ op_b(B_i). Small
// GEMMs are launch bound: one strided batch. cuBLAS picks slower kernels for
// large strided batches (H=512: 0.9x), so those loop plain GEMMs.
static void muon_gemm(cublasOperation_t op_a, cublasOperation_t op_b,
        int M, int N, int K, precision_t* A, long sa, precision_t* B, long sb,
        precision_t* C, long sc, int count, float alpha, cudaStream_t stream) {
    int lda = (op_a == CUBLAS_OP_N) ? K : M;
    int ldb = (op_b == CUBLAS_OP_N) ? N : K;
    float beta = 0.0f;
    cublasSetStream(g_cublas_handle, stream);
    cublasSetWorkspace(g_cublas_handle, g_cublas_workspace, CUBLAS_WS_BYTES);
    if ((long)M * N * K <= (1L << 26)) {
        cublasGemmStridedBatchedEx(g_cublas_handle, op_b, op_a, N, M, K, &alpha,
            B, CUBLAS_PRECISION, ldb, sb, A, CUBLAS_PRECISION, lda, sa, &beta,
            C, CUBLAS_PRECISION, N, sc, count, CUBLAS_COMPUTE, CUBLAS_GEMM_DEFAULT);
        return;
    }
    for (int i = 0; i < count; i++) {
        cublasGemmEx(g_cublas_handle, op_b, op_a, N, M, K, &alpha,
            B + i * sb, CUBLAS_PRECISION, ldb, A + i * sa, CUBLAS_PRECISION, lda,
            &beta, C + i * sc, CUBLAS_PRECISION, N, CUBLAS_COMPUTE, CUBLAS_GEMM_DEFAULT);
    }
}

// Updates fp32 master weights and refreshes the precision_t params.
void muon_step(Muon* m, Float weights, Prec params, Prec grads,
        float max_grad_norm, cudaStream_t stream) {
    int n_grad = (int)numel(grads.shape);
    int nparts = min((int)grid_size(n_grad), 256);
    // Grads mirror the param layout (incl. alignment padding): offsets come from
    // the param data pointers. The head regs are contiguous: one flat Adam range.
    precision_t* param_base = (precision_t*)m->param_alloc->mem;
    long adam_lo = 0, adam_hi = 0;
    if (m->head_hi > m->head_lo) {
        AllocEntry& last = m->param_alloc->regs[m->head_hi - 1];
        adam_lo = (precision_t*)*m->param_alloc->regs[m->head_lo].data_ptr - param_base;
        adam_hi = (precision_t*)*last.data_ptr - param_base + numel(last.shape);
    }
    muon_sum_sq_partials<<<nparts, 256, 0, stream>>>(
        m->norm_partials, grads.data, n_grad);
    muon_clip_nesterov<<<grid_size(n_grad), BLOCK_SIZE, 0, stream>>>(
        m->mb.data, grads.data, m->norm_partials, nparts,
        max_grad_norm, 1e-6f, (float)m->momentum, n_grad,
        m->vb.data, adam_lo, adam_hi, m->adam_t, m->head_lr_ratio);

    // Per group: normalize momentum into x_buf, then 5 NS steps ping-pong
    // x_buf -> grads -> ... -> grads, so the scaled update lands in the flat
    // grads. 1D params keep their Nesterov update in place (scale 1). Small
    // groups instead join one muon_ns_small launch.
    MuonSmall small;
    small.blk0[0] = 0;
    int n_small = 0;
    for (int gi = 0; gi < m->num_groups; gi++) {
        MuonGroup* g = &m->groups[gi];
        long off = (precision_t*)*m->param_alloc->regs[g->reg].data_ptr - param_base;
        precision_t* gc = grads.data + off;
        long R = g->R, C = g->C, M = min(R, C), n = R * C, mm = M * M;
        int count = g->count;
        if (g->small) {
            for (int i = 0; i < count; i++) {
                small.off[n_small] = off + i * n;
                small.R[n_small] = (int)R;
                small.C[n_small] = (int)C;
                small.blk0[n_small + 1] = small.blk0[n_small] + g->blocks;
                n_small++;
            }
            continue;
        }
        bool tall = R > C;
        int np = min((int)grid_size(n), 256);
        muon_sum_sq_partials<<<dim3(np, count), 256, 0, stream>>>(
            m->norm_partials, gc, n);
        muon_l2_normalize<<<dim3(grid_size(n), count), BLOCK_SIZE, 0, stream>>>(
            m->x_buf.data, gc, m->norm_partials, np, 1e-7f, n);

        // 5 steps. 4 = you break it.
        precision_t* gram = m->gram.data;
        precision_t* poly = m->gram_buf.data;
        for (int i = 0; i < 5; ++i) {
            precision_t* src = (i % 2 == 0) ? m->x_buf.data : gc;
            precision_t* dst = (i % 2 == 0) ? gc : m->x_buf.data;
            if (tall) {
                muon_gemm(CUBLAS_OP_T, CUBLAS_OP_N, M, M, R, src, n, src, n,
                    gram, mm, count, 1.0f, stream);
            } else {
                muon_gemm(CUBLAS_OP_N, CUBLAS_OP_T, M, M, C, src, n, src, n,
                    gram, mm, count, 1.0f, stream);
            }
            muon_gemm(CUBLAS_OP_N, CUBLAS_OP_N, M, M, M, gram, mm, gram, mm,
                poly, mm, count, (float)ns_coeffs[i][2], stream);
            muon_ns_poly<<<grid_size(count * mm), BLOCK_SIZE, 0, stream>>>(
                poly, gram, (float)ns_coeffs[i][0], (float)ns_coeffs[i][1],
                M, count * mm);
            // Last step folds in the aspect scale.
            float alpha = (i == 4) ? sqrtf(fmaxf(1.0f, (float)R / (float)C)) : 1.0f;
            if (tall) {
                muon_gemm(CUBLAS_OP_N, CUBLAS_OP_N, R, C, C, src, n, poly, mm,
                    dst, n, count, alpha, stream);
            } else {
                muon_gemm(CUBLAS_OP_N, CUBLAS_OP_N, R, C, R, poly, mm, src, n,
                    dst, n, count, alpha, stream);
            }
        }
    }
    if (n_small > 0) {
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 3; j++) {
                small.coef[i][j] = (float)ns_coeffs[i][j];
            }
        }
        if (m->ns_split) {
            void* args[] = {&grads.data, &m->ns_part, &small};
            cudaLaunchCooperativeKernel((void*)muon_ns_small<true>, m->ns_blocks,
                MUON_SMALL_THREADS, args, m->small_smem, stream);
        } else {
            muon_ns_small<false><<<m->ns_blocks, MUON_SMALL_THREADS, m->small_smem,
                stream>>>(grads.data, m->ns_part, small);
        }
    }
    muon_weight_update<<<grid_size(n_grad), BLOCK_SIZE, 0, stream>>>(
        weights.data, params.data, grads.data, m->lr, 0.0f, n_grad, m->adam_t);
}

// Train layout is (B, T). Views are sliced each mb; scratch is allocated.
struct TrainGraph {
    Prec mb_state;       // view into train_state (L, A, H); read with agent_off
    Prec mb_obs;         // view (B, T, input_size)
    Float mb_actions;    // view (B, T, num_atns)
    Prec mb_logprobs;    // view (B, T)
    Prec mb_terminals;   // view (B, T)
    Prec mb_rewards;     // view (B, T)
    Prec mb_advantages;  // scratch
    Prec mb_values;      // view: frozen rollout V (vf-clip)
    Prec mb_returns;     // view: aliases mb_gae_v after GAE (V+A)
    Prec mb_action_mask; // view (B, T, mask_size)
    Prec mb_imp;         // scratch
    Prec mb_gae_v;       // scratch: live V in, overwritten with returns
};

void register_train_buffers(TrainGraph& bufs, Allocator* alloc, int B, int T) {
    bufs = (TrainGraph){
        .mb_advantages =    {.shape = {B, T}},
        .mb_imp =           {.shape = {B, T}},
        .mb_gae_v =         {.shape = {B, T}},
    };
    alloc_register(alloc, &bufs.mb_advantages);
    alloc_register(alloc, &bufs.mb_imp);
    alloc_register(alloc, &bufs.mb_gae_v);
}

// TODO: test whether these finite/clamp guards improve continuous-control stability
// or just hide bad logits/actions.
__device__ __forceinline__ float finite_or_clamp(float x, float lo, float hi) {
    if (isnan(x)) {
        return 0.0f;
    }
    if (isinf(x)) {
        return x > 0.0f ? hi : lo;
    }
    return fminf(hi, fmaxf(lo, x));
}

__device__ __forceinline__ float safe_continuous_mean(const precision_t* logits, int idx) {
    return finite_or_clamp(to_float(logits[idx]), -1.0e6f, 1.0e6f);
}

__device__ __forceinline__ float safe_continuous_logstd(const precision_t* logstd, int idx) {
    return finite_or_clamp(to_float(logstd[idx]), -20.0f, 2.0f);
}

enum LossIdx {
    LOSS_PG = 0, LOSS_VF = 1, LOSS_ENT = 2, LOSS_TOTAL = 3,
    LOSS_OLD_APPROX_KL = 4, LOSS_APPROX_KL = 5, LOSS_CLIPFRAC = 6,
    LOSS_IMP = 7,
    LOSS_N = 8, NUM_LOSSES = 9,
};

constexpr int PPO_THREADS = 256;

// Per-env from ENV_HEADER (ocean/<env>/<env>.h).
#ifndef NUM_ATNS
#error "ENV_HEADER must #define NUM_ATNS (number of action heads)"
#endif
#ifndef ACT_SIZES
#error "ENV_HEADER must #define ACT_SIZES { ... } (classes per head)"
#endif
// Exact max head width for this env build — discrete logit cache size.
constexpr int ppo_max_head_classes() {
    constexpr int s[] = ACT_SIZES;
    int m = 0;
    for (int i = 0; i < NUM_ATNS; i++) {
        if (s[i] > m) {
            m = s[i];
        }
    }
    return m > 0 ? m : 1;
}
constexpr int PPO_MAX_HEAD_A = ppo_max_head_classes();

// Fused loss function. PPO clipped loss + value + entropy
// buffers + args are quite complex. We do the entire
// forward + backwards pass for the full loss function in one kernel
struct PPOGraphArgs {
    precision_t* imp;
    const float* actions;
    const precision_t* old_logprobs;
    const precision_t* advantages;
    const precision_t* values;
    const precision_t* returns;
};

struct PPOKernelArgs {
    float* grad_logits;
    float* grad_logstd; // For continuous actions
    float* grad_values_pred;
    const precision_t* logits;
    const precision_t* logstd; // Continuous only
    const precision_t* values_pred;
    const int* act_sizes;
    const precision_t* action_mask; // (N, T, A_total); always present
    int num_atns;
    float clip_coef, vf_clip_coef, vf_coef;
    const float* ent_coef;  // device ptr — host by-value bakes into CUDA graphs
    int T_seq, A_total, N;
    bool is_continuous;
};

struct PPOBufs {
    Float grad_logits, grad_values, grad_logstd;
    Float ppo_partials;
    float* ent_coef;     // device scalar (graphs cannot bake host by-value)
};

void register_ppo_buffers(PPOBufs& bufs, Allocator* alloc, int N, int T, int A_total, bool is_continuous) {
    long total = (long)N * T;
    int ppo_grid = ((int)total + PPO_THREADS - 1) / PPO_THREADS;
    bufs = (PPOBufs){
        .grad_logits = {.shape = {N, T, A_total}},
        .grad_values = {.shape = {N, T, 1}},
        .grad_logstd = {.shape = {N, T, A_total}},
        .ppo_partials = {.shape = {ppo_grid * LOSS_N}},
        .ent_coef = NULL,
    };
    alloc_register(alloc, &bufs.grad_logits);
    alloc_register(alloc, &bufs.grad_values);
    if (is_continuous) {
        alloc_register(alloc, &bufs.grad_logstd);
    }
    alloc_register(alloc, &bufs.ppo_partials);
    cudaMalloc((void**)&bufs.ent_coef, sizeof(float));
}

// Discrete only. mask is always present (env mask or synthetic all-ones).
__device__ __forceinline__ float load_logit_masked(
        const precision_t* __restrict__ logits, int logits_base,
        int logits_offset, int a,
        const precision_t* __restrict__ mask, int mask_base) {
    float l = to_float(logits[logits_base + logits_offset + a]);
    float m = to_float(mask[mask_base + logits_offset + a]);
    if (m == 0.0f) {
        l = -1e4f;
    }
    return l;
}

// Shared by sample_logits + ppo_loss. Fills cache[0..A) (sized PPO_MAX_HEAD_A).
__device__ __forceinline__ float ppo_discrete_logsumexp(
        const precision_t* __restrict__ logits, int logits_base,
        int logits_offset, int A,
        const precision_t* __restrict__ mask, int mask_base,
        float* __restrict__ cache) {
    float max_logit = -INFINITY;
    float sum = 0.0f;
    for (int a = 0; a < A; ++a) {
        float l = load_logit_masked(
            logits, logits_base, logits_offset, a, mask, mask_base);
        cache[a] = l;
        if (l > max_logit) {
            sum *= __expf(max_logit - l);
            max_logit = l;
        }
        sum += __expf(l - max_logit);
    }
    return max_logit + __logf(sum);
}

__device__ __forceinline__ void ppo_continuous_head(
        float mean, float log_std, float action,
        float* out_logp, float* out_entropy) {
    constexpr float HALF_LOG_2PI = 0.9189385332046727f;
    constexpr float HALF_1_PLUS_LOG_2PI = 1.4189385332046727f;
    float std = __expf(log_std);
    float normalized = (action - mean) / std;
    *out_logp = -0.5f * normalized * normalized - HALF_LOG_2PI - log_std;
    *out_entropy = HALF_1_PLUS_LOG_2PI + log_std;
}

// After decoder GEMM, before GAE. One logit walk: live V, ρ, logp=logit-lse.
// logp is written into grad_logits (overwritten with grads after GAE).
__global__ void cache_imp_and_v(
        Prec dec_out,
        const float* __restrict__ actions,
        const precision_t* __restrict__ old_logprobs,
        const precision_t* __restrict__ action_mask,
        Prec logstd,
        const int* __restrict__ act_sizes,
        precision_t* __restrict__ imp_out,
        precision_t* __restrict__ value_out,
        float* __restrict__ logps,
        float* __restrict__ new_lp_out) {
    int NT = (int)dec_out.shape[0] * (int)dec_out.shape[1];
    int fused_cols = (int)dec_out.shape[2];
    int A_total = fused_cols - 1;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= NT) {
        return;
    }
    int logits_base = idx * fused_cols;
    int at_base = idx * A_total;
    precision_t* logits = dec_out.data;
    value_out[idx] = logits[logits_base + A_total];

    float new_lp = 0.0f;
    if (logstd.data) {
        for (int h = 0; h < NUM_ATNS; ++h) {
            float lp, ent;
            ppo_continuous_head(
                safe_continuous_mean(logits, logits_base + h),
                safe_continuous_logstd(logstd.data, h),
                actions[idx * NUM_ATNS + h], &lp, &ent);
            new_lp += lp;
        }
    } else {
#ifdef PUFFER_NETHACK
        int verb = (int)actions[idx * NUM_ATNS];
#endif
        int logits_offset = 0;
        for (int h = 0; h < NUM_ATNS; ++h) {
            int A = act_sizes[h];
            float cache[PPO_MAX_HEAD_A];
            float lse = ppo_discrete_logsumexp(
                logits, logits_base, logits_offset, A,
                action_mask, at_base, cache);
            for (int a = 0; a < A; ++a) {
                logps[at_base + logits_offset + a] = cache[a] - lse;
            }
#ifdef PUFFER_NETHACK
            int used = nethack_head_used(verb, h);
            if (used) {
                int act = (int)actions[idx * NUM_ATNS + h];
                if (h == 0) {
                    float mix = 1.0f;
                    new_lp += nethack_verb_train_logp(
                        action_mask + at_base + logits_offset, A,
                        cache[act] - lse, &mix);
                } else {
                    new_lp += cache[act] - lse;
                }
            }
#else
            new_lp += cache[(int)actions[idx * NUM_ATNS + h]] - lse;
#endif
            logits_offset += A;
        }
    }
    new_lp_out[idx] = new_lp;
    imp_out[idx] = from_float(__expf(new_lp - to_float(old_logprobs[idx])));
}

Prec arch_forward_train(Arch* p, Weights& w,
        Activations& activations, Prec x,
        Prec state, Prec terminals, int agent_off,
        TrainGraph& g, Prec logstd, int* act_sizes,
        float* logps, float* new_lp, cudaStream_t stream) {
    int B = x.shape[0], TT = x.shape[1];
    Prec h = p->encoder.forward(w.encoder,
        activations.encoder, *puf_squeeze(&x, 0), stream);
    h = p->network.forward_train(w.network, *puf_unsqueeze(&h, 0, B, TT),
        state, terminals, activations.network, agent_off, stream);
    Prec dec_out = p->decoder.forward(
        w.decoder, activations.decoder, *puf_squeeze(&h, 0), stream);
    Prec dec = *puf_unsqueeze(&dec_out, 0, B, TT);
    cache_imp_and_v<<<grid_size(B * TT), BLOCK_SIZE, 0, stream>>>(
        dec, g.mb_actions.data, g.mb_logprobs.data, g.mb_action_mask.data,
        logstd, act_sizes, g.mb_imp.data, g.mb_gae_v.data, logps, new_lp);
    return dec;
}

__global__ void ppo_loss_compute(
        float* __restrict__ ppo_partials,
        PPOKernelArgs a, PPOGraphArgs g) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    int total_elements = a.N * a.T_seq;
    float inv_NT = 1.0f / float(total_elements);

    __shared__ float block_losses[LOSS_N][PPO_THREADS];
    for (int c = 0; c < LOSS_N; c++) {
        block_losses[c][tid] = 0.0f;
    }

    if (idx < total_elements) {
        int nt = idx;
        int logits_base = nt * (a.A_total + 1);
        int at_base = nt * a.A_total;  // logits-grad + mask base (A_total cols)

        float adv_for_pg = to_float(g.advantages[nt]);
        float val = to_float(g.values[nt]);
        float ret = to_float(g.returns[nt]);
        float val_pred = to_float(a.values_pred[logits_base]);
        float ent_coef = *a.ent_coef;
        float d_entropy_term = inv_NT * (-ent_coef);
        float logratio = a.grad_values_pred[nt] - to_float(g.old_logprobs[nt]);
        float ratio = __expf(logratio);

        // Value loss + gradient: 0.5 * max((v-r)^2, (v_clip-r)^2).
        // When clipped term wins, v_clip is constant in val_pred → grad 0.
        float v_error = val_pred - val;
        float v_clipped = val + fmaxf(-a.vf_clip_coef, fminf(a.vf_clip_coef, v_error));
        float v_loss_unclipped = (val_pred - ret) * (val_pred - ret);
        float v_loss_clipped = (v_clipped - ret) * (v_clipped - ret);
        float v_loss = 0.5f * fmaxf(v_loss_unclipped, v_loss_clipped);
        float d_val_pred = (v_loss_clipped > v_loss_unclipped) ? 0.0f : (val_pred - ret);
        a.grad_values_pred[nt] = inv_NT * a.vf_coef * d_val_pred;
        float clip_lo = 1.0f - a.clip_coef;
        float clip_hi = 1.0f + a.clip_coef;
        float ratio_clipped = fmaxf(clip_lo, fminf(clip_hi, ratio));
        float wa = -adv_for_pg;
        float pg_loss1 = wa * ratio;
        float pg_loss2 = wa * ratio_clipped;
        float pg_loss = fmaxf(pg_loss1, pg_loss2);
        float d_ratio = wa * inv_NT;
        if (pg_loss2 > pg_loss1 && (ratio <= clip_lo || ratio >= clip_hi)) {
            d_ratio = 0.0f;
        }
        float d_new_logp = d_ratio * ratio;
        float total_entropy = 0.0f;

        if (a.is_continuous) {
            for (int h = 0; h < a.num_atns; ++h) {
                float mean = safe_continuous_mean(a.logits, logits_base + h);
                float c_logstd = safe_continuous_logstd(a.logstd, h);
                float c_action = finite_or_clamp(
                    g.actions[nt * a.num_atns + h], -1.0e6f, 1.0e6f);
                float lp, ent;
                ppo_continuous_head(mean, c_logstd, c_action, &lp, &ent);
                total_entropy += ent;
                float std = __expf(c_logstd);
                float var = std * std;
                float diff = c_action - mean;
                a.grad_logits[at_base + h] = d_new_logp * diff / var;
                a.grad_logstd[nt * a.num_atns + h] =
                    d_new_logp * (diff * diff / var - 1.0f) + d_entropy_term;
            }
        } else {
#ifdef PUFFER_NETHACK
            int verb = (int)g.actions[nt * a.num_atns];
            float verb_mix_scale = 1.0f;
#endif
            int logits_offset = 0;
            for (int h = 0; h < a.num_atns; ++h) {
                int A = a.act_sizes[h];
#ifdef PUFFER_NETHACK
                if (!nethack_head_used(verb, h)) {
                    for (int j = 0; j < A; ++j) {
                        a.grad_logits[at_base + logits_offset + j] = 0.0f;
                    }
                    logits_offset += A;
                    continue;
                }
#endif
                int act = (int)g.actions[nt * a.num_atns + h];
                float ent = 0.0f;
                for (int j = 0; j < A; ++j) {
                    float logp = a.grad_logits[at_base + logits_offset + j];
                    ent -= __expf(logp) * logp;
                }
                total_entropy += ent;
#ifdef PUFFER_NETHACK
                float d_logp = d_new_logp;
                if (h == 0) {
                    nethack_verb_train_logp(
                        a.action_mask + at_base + logits_offset, A,
                        a.grad_logits[at_base + act],
                        &verb_mix_scale);
                    d_logp *= verb_mix_scale;
                }
#else
                float d_logp = d_new_logp;
#endif
                for (int j = 0; j < A; ++j) {
                    float logp = a.grad_logits[at_base + logits_offset + j];
                    float p = __expf(logp);
                    a.grad_logits[at_base + logits_offset + j] =
                        ((j == act ? 1.0f : 0.0f) - p) * d_logp
                        + d_entropy_term * p * (-ent - logp);
                }
                logits_offset += A;
            }
        }

        float thread_loss = (pg_loss + a.vf_coef * v_loss
            - ent_coef * total_entropy) * inv_NT;

        block_losses[LOSS_PG][tid] = pg_loss * inv_NT;
        block_losses[LOSS_VF][tid] = v_loss * inv_NT;
        block_losses[LOSS_ENT][tid] = total_entropy * inv_NT;
        block_losses[LOSS_TOTAL][tid] = thread_loss;
        block_losses[LOSS_OLD_APPROX_KL][tid] = (-logratio) * inv_NT;
        block_losses[LOSS_APPROX_KL][tid] = ((ratio - 1.0f) - logratio) * inv_NT;
        block_losses[LOSS_CLIPFRAC][tid] =
            (fabsf(ratio - 1.0f) > a.clip_coef ? 1.0f : 0.0f) * inv_NT;
        block_losses[LOSS_IMP][tid] = ratio * inv_NT;
    }

    block_reduce_sum(&block_losses[0][0], &ppo_partials[blockIdx.x * LOSS_N],
        tid, PPO_THREADS, LOSS_N);
}

// Deterministic reduction of per-block PPO loss partials + count increment
__global__ void ppo_loss_reduce(
        float* __restrict__ losses_acc,
        const float* __restrict__ partials,
        int num_blocks) {
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int b = 0; b < num_blocks; b++) {
        sum += partials[b * LOSS_N + tid];
    }
    losses_acc[tid] += sum;
    if (tid == 0) {
        losses_acc[LOSS_N] += 1.0f;
    }
}

void ppo_loss_fwd_bwd(
        Prec& dec_out,    // (N, T, fused_cols) - fused logits+value from decoder
        Prec& logstd,     // continuous logstd or empty
        TrainGraph& graph,
        int* act_sizes, float* losses_acc,
        float clip_coef, float vf_clip_coef, float vf_coef, const float* ent_coef,
        PPOBufs& bufs, bool is_continuous,
        cudaStream_t stream) {
    int N = dec_out.shape[0], T = dec_out.shape[1], fused_cols = dec_out.shape[2];
    int A_total = fused_cols - 1;  // last column is value
    int total = N * T;
    int ppo_grid = (total + PPO_THREADS - 1) / PPO_THREADS;

    PPOGraphArgs graph_args = {
        .imp = graph.mb_imp.data,
        .actions = graph.mb_actions.data,
        .old_logprobs = graph.mb_logprobs.data,
        .advantages = graph.mb_advantages.data,
        .values = graph.mb_values.data,
        .returns = graph.mb_returns.data,
    };

    PPOKernelArgs args = {
        .grad_logits = bufs.grad_logits.data,
        .grad_logstd = is_continuous ? bufs.grad_logstd.data : NULL,
        .grad_values_pred = bufs.grad_values.data,
        .logits = dec_out.data,
        .logstd = is_continuous ? logstd.data : NULL,
        .values_pred = dec_out.data + A_total,
        .act_sizes = act_sizes,
        .action_mask = graph.mb_action_mask.data,
        .num_atns = NUM_ATNS,
        .clip_coef = clip_coef, .vf_clip_coef = vf_clip_coef,
        .vf_coef = vf_coef,
        .ent_coef = ent_coef,
        .T_seq = T, .A_total = A_total, .N = N,
        .is_continuous = is_continuous,
    };
    ppo_loss_compute<<<ppo_grid, PPO_THREADS, 0, stream>>>(
            bufs.ppo_partials.data, args, graph_args);
    // Loss sums only feed logging: off the critical path.
    ppo_loss_reduce<<<1, LOSS_N, 0, puf_fork(stream)>>>(
        losses_acc, bufs.ppo_partials.data, ppo_grid);
}

// Puffer advantage (GAE + V-trace ρ/c clips). One thread per row.
// 16B segment I/O (H % ADV_VEC_WIDTH == 0). Bench: tests/bench_puff_advantage.cu
constexpr int ADV_VEC_WIDTH = 16 / (int)sizeof(precision_t);

// Wide load/store of one ADV_VEC_WIDTH segment into float regs.
__device__ __forceinline__ void adv_ld(const precision_t* p, float* o) {
    if constexpr (sizeof(precision_t) == sizeof(float)) {
        float4 v = *(const float4*)p;
        o[0] = v.x; o[1] = v.y; o[2] = v.z; o[3] = v.w;
    } else {
        uint4 u = *(const uint4*)p;
        auto* b = (const __nv_bfloat16*)&u;
        #pragma unroll
        for (int i = 0; i < 8; i++) o[i] = __bfloat162float(b[i]);
    }
}
__device__ __forceinline__ void adv_st(precision_t* p, const float* o) {
    if constexpr (sizeof(precision_t) == sizeof(float)) {
        *(float4*)p = make_float4(o[0], o[1], o[2], o[3]);
    } else {
        __nv_bfloat16 b[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) b[i] = __float2bfloat16(o[i]);
        *(uint4*)p = *(const uint4*)b;
    }
}

// GAE / full truncated IS (V-trace ρ/c): ρ̄ on δ, c̄ on λ product.
// Same R_{t+1},D_{t+1} indexing as classic puffer_advantage.
__global__ void puff_advantage(const precision_t* values,
        const precision_t* rewards, const precision_t* dones,
        const precision_t* importance, precision_t* advantages,
        precision_t* returns,
        float gamma, float lambda, float rho_clip, float c_clip,
        int num_steps, int horizon) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_steps) return;
    int off = row * horizon;
    float lastlam = 0.f;
    float next_v = to_float(values[off + horizon - 1]);
    float next_d = to_float(dones[off + horizon - 1]);
    float next_r = to_float(rewards[off + horizon - 1]);

    for (int seg = horizon / ADV_VEC_WIDTH - 1; seg >= 0; seg--) {
        int base = off + seg * ADV_VEC_WIDTH;
        float v[ADV_VEC_WIDTH], r[ADV_VEC_WIDTH], d[ADV_VEC_WIDTH], imp[ADV_VEC_WIDTH];
        float adv[ADV_VEC_WIDTH] = {};
        float ret[ADV_VEC_WIDTH];
        adv_ld(values + base, v);
        adv_ld(rewards + base, r);
        adv_ld(dones + base, d);
        if (importance) {
            adv_ld(importance + base, imp);
        } else {
            #pragma unroll
            for (int i = 0; i < ADV_VEC_WIDTH; i++) {
                imp[i] = 1.f;
            }
        }
        // Last index H-1 left 0. First seg starts at width-2.
        int i0 = (seg + 1 == horizon / ADV_VEC_WIDTH) ? ADV_VEC_WIDTH - 2 : ADV_VEC_WIDTH - 1;
        #pragma unroll
        for (int i = i0; i >= 0; i--) {
            float nnt = 1.f - next_d;
            float rho_t = fminf(imp[i], rho_clip);
            float c_t = fminf(imp[i], c_clip);
            float delta = rho_t * (next_r + gamma * next_v * nnt - v[i]);
            lastlam = delta + gamma * lambda * c_t * lastlam * nnt;
            adv[i] = lastlam;
            next_v = v[i]; next_d = d[i]; next_r = r[i];
        }
        #pragma unroll
        for (int i = 0; i < ADV_VEC_WIDTH; i++) {
            ret[i] = v[i] + adv[i];
        }
        adv_st(advantages + base, adv);
        adv_st(returns + base, ret);
    }
}
