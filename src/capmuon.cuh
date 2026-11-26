#pragma once
// CapMuon: O(d_in) Capped Input-Geometry Preconditioning for Muon.
//
// Tracks the top-1 input eigenmode (v1, lambda1) per 2D layer via an exact-Rayleigh
// EMA power-iteration sketch and applies a one-sided cap S = I - (1 - s1) v1 v1^T
// with s1 = min(1, (c * mean(lambda) / lambda1)^p) <= 1:
//
//     dW = -lr * polar(Z S) S
//
// Because S <= I, tail eigenvalues are never amplified. Inner polar(Z S) redirects
// orthogonal update energy away from dominant input spikes; outer S bounds activation
// step energy. 100% CUDA-graph compatible (zero host syncs, zero dense d x d buffers).

constexpr int MAX_CAP_LAYERS = 16;
constexpr int MAX_CAP_SMEM_BYTES = 48 * 1024;

struct CapLayer {
    bool enabled;
    int kind;        // 0=encoder, 1=mingru, 2=decoder
    int kind_idx;    // layer index within kind (for mingru)
    int d;           // in_dim (columns C)
    int out_dim;     // out_dim (rows R)
    long param_off;  // element offset of weight in flat param/grad buffer
    float ema_gamma; // EMA decay (default 0.9)
    float cap_c;     // threshold ratio c (default 1.0)
    float cap_p;     // cap power p (default 0.5)

    // Device buffers (all O(d) or O(out_dim))
    float* v1;       // [d] unit top eigenvector
    float* acc_mean; // [d] scratch batch sum x (for step-0 mean init)
    float* acc_gv;   // [d] scratch batch sum x*(x.v)
    float* acc_sc;   // [2] scratch {sum ||x||^2, sum (x.v)^2}
    float* state;    // [4] {lam1, tr_ema, s1, steps}
    float* zv;       // [out_dim] scratch row projections Z*v or Q*v
};

struct CapManager {
    int num_layers;
    CapLayer layers[MAX_CAP_LAYERS];
};

inline void cap_layer_init(CapLayer* c, int kind, int kind_idx,
        int d, int out_dim, long param_off,
        float ema_gamma, float cap_c, float cap_p) {
    memset(c, 0, sizeof(*c));
    if (d <= 0 || out_dim <= 0 || (3 * d * (int)sizeof(float)) > MAX_CAP_SMEM_BYTES) {
        return;
    }
    c->enabled = true;
    c->kind = kind;
    c->kind_idx = kind_idx;
    c->d = d;
    c->out_dim = out_dim;
    c->param_off = param_off;
    c->ema_gamma = ema_gamma;
    c->cap_c = cap_c;
    c->cap_p = cap_p;

    cudaMalloc((void**)&c->v1, d * sizeof(float));
    cudaMalloc((void**)&c->acc_mean, d * sizeof(float));
    cudaMalloc((void**)&c->acc_gv, d * sizeof(float));
    cudaMalloc((void**)&c->acc_sc, 2 * sizeof(float));
    cudaMalloc((void**)&c->state, 4 * sizeof(float));
    cudaMalloc((void**)&c->zv, out_dim * sizeof(float));

    cudaMemset(c->v1, 0, d * sizeof(float));
    cudaMemset(c->acc_mean, 0, d * sizeof(float));
    cudaMemset(c->acc_gv, 0, d * sizeof(float));
    cudaMemset(c->acc_sc, 0, 2 * sizeof(float));
    float init_state[4] = {0.0f, 0.0f, 1.0f, 0.0f};
    cudaMemcpy(c->state, init_state, 4 * sizeof(float), cudaMemcpyHostToDevice);
    float v0 = 1.0f;
    cudaMemcpy(c->v1, &v0, sizeof(float), cudaMemcpyHostToDevice);
}

inline CapLayer* cap_mgr_find(CapManager* mgr, long param_off, int R, int C) {
    for (int i = 0; i < mgr->num_layers; i++) {
        CapLayer* c = &mgr->layers[i];
        if (c->enabled && c->param_off == param_off && c->out_dim == R && c->d == C) {
            return c;
        }
    }
    return NULL;
}

// Single-pass warp-per-row coalesced accumulation of sum_x, sum_x*(x.v),
// sum ||x||^2, and sum (x.v)^2.
__global__ void cap_sketch_accum_kernel(
        const precision_t* __restrict__ X, int B, int d,
        const float* __restrict__ v1,
        float* __restrict__ acc_mean, float* __restrict__ acc_gv,
        float* __restrict__ acc_sc) {
    extern __shared__ float smem[];
    float* s_v = smem;
    float* s_mean = smem + d;
    float* s_gv = smem + 2 * d;
    __shared__ float s_sc[2];

    int tid = threadIdx.x;
    for (int j = tid; j < d; j += blockDim.x) {
        s_v[j] = v1[j];
        s_mean[j] = 0.0f;
        s_gv[j] = 0.0f;
    }
    if (tid < 2) s_sc[tid] = 0.0f;
    __syncthreads();

    int warp = tid >> 5;
    int lane = tid & 31;
    int nwarps = blockDim.x >> 5;

    for (int r = blockIdx.x * nwarps + warp; r < B; r += gridDim.x * nwarps) {
        const precision_t* row = X + (long)r * d;
        float y = 0.0f;
        float tr = 0.0f;
        for (int j = lane; j < d; j += 32) {
            float xj = to_float(row[j]);
            y += xj * s_v[j];
            tr += xj * xj;
        }
        for (int off = 16; off > 0; off >>= 1) {
            y += __shfl_down_sync(0xffffffff, y, off);
            tr += __shfl_down_sync(0xffffffff, tr, off);
        }
        y = __shfl_sync(0xffffffff, y, 0);
        if (lane == 0) {
            atomicAdd(&s_sc[0], tr);
            atomicAdd(&s_sc[1], y * y);
        }
        for (int j = lane; j < d; j += 32) {
            float xj = to_float(row[j]);
            atomicAdd(&s_mean[j], xj);
            atomicAdd(&s_gv[j], xj * y);
        }
    }
    __syncthreads();

    for (int j = tid; j < d; j += blockDim.x) {
        atomicAdd(&acc_mean[j], s_mean[j]);
        atomicAdd(&acc_gv[j], s_gv[j]);
    }
    if (tid < 2) {
        atomicAdd(&acc_sc[tid], s_sc[tid]);
    }
}

// Single-block finalize: updates Rayleigh-EMA (v1, lam1, tr_ema, s1) and zeroes acc_*.
__global__ void cap_sketch_finalize_kernel(
        int B, int d, float ema_gamma, float cap_c, float cap_p,
        float* __restrict__ v1, float* __restrict__ acc_mean,
        float* __restrict__ acc_gv, float* __restrict__ acc_sc,
        float* __restrict__ state) {
    __shared__ float s_buf[256];
    int tid = threadIdx.x;
    float invB = 1.0f / (float)B;

    float prev_lam1 = state[0];
    float prev_tr   = state[1];
    float steps     = state[3];
    float b_tr      = acc_sc[0] * invB;
    float b_lam1    = acc_sc[1] * invB;

    if (steps < 0.5f) {
        float mn2 = 0.0f;
        for (int j = tid; j < d; j += blockDim.x) {
            float m = acc_mean[j] * invB;
            acc_mean[j] = m;
            mn2 += m * m;
        }
        s_buf[tid] = mn2;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (tid < s) s_buf[tid] += s_buf[tid + s];
            __syncthreads();
        }
        float mn = sqrtf(s_buf[0]);
        for (int j = tid; j < d; j += blockDim.x) {
            v1[j] = (mn > 1e-12f) ? (acc_mean[j] / mn) : ((j == 0) ? 1.0f : 0.0f);
            acc_mean[j] = 0.0f;
            acc_gv[j] = 0.0f;
        }
        if (tid == 0) {
            state[0] = b_lam1;
            state[1] = b_tr;
            state[2] = 1.0f;
            state[3] = 1.0f;
            acc_sc[0] = 0.0f;
            acc_sc[1] = 0.0f;
        }
        return;
    }

    float g = (steps < 1.5f) ? 0.0f : ema_gamma;
    float lam1 = g * prev_lam1 + (1.0f - g) * b_lam1;
    float tr_ema = g * prev_tr + (1.0f - g) * b_tr;

    float u_norm2 = 0.0f;
    for (int j = tid; j < d; j += blockDim.x) {
        float gv = acc_gv[j] * invB;
        float uj = g * prev_lam1 * v1[j] + (1.0f - g) * gv;
        acc_gv[j] = uj;
        u_norm2 += uj * uj;
    }
    s_buf[tid] = u_norm2;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) s_buf[tid] += s_buf[tid + s];
        __syncthreads();
    }
    float un = sqrtf(s_buf[0]);
    for (int j = tid; j < d; j += blockDim.x) {
        if (un > 1e-12f) v1[j] = acc_gv[j] / un;
        acc_mean[j] = 0.0f;
        acc_gv[j] = 0.0f;
    }
    if (tid == 0) {
        float mean_eig = tr_ema / (float)d;
        float thresh = cap_c * mean_eig;
        float s1 = 1.0f;
        if (thresh > 1e-12f && lam1 > thresh) {
            s1 = powf(thresh / lam1, cap_p);
        }
        state[0] = lam1;
        state[1] = tr_ema;
        state[2] = s1;
        state[3] = steps + 1.0f;
        acc_sc[0] = 0.0f;
        acc_sc[1] = 0.0f;
    }
}

inline void cap_sketch_update(CapLayer* c, Prec saved_input, cudaStream_t stream) {
    if (!c->enabled) return;
    int d = c->d;
    long tot = numel(saved_input.shape);
    if (d <= 0 || tot <= 0 || (tot % d) != 0) return;
    int B = (int)(tot / d);
    int blocks = min((B + 3) / 4, 256);
    size_t smem = (size_t)(3 * d) * sizeof(float);
    cap_sketch_accum_kernel<<<blocks, 128, smem, stream>>>(
        saved_input.data, B, d, c->v1, c->acc_mean, c->acc_gv, c->acc_sc);
    cap_sketch_finalize_kernel<<<1, 256, 0, stream>>>(
        B, d, c->ema_gamma, c->cap_c, c->cap_p,
        c->v1, c->acc_mean, c->acc_gv, c->acc_sc, c->state);
}

// Fused rank-1 inner S projection + exact Frobenius norm with fast s1==1.0 skip.
__global__ void cap_r1_inner_proj_norm_kernel(
        const precision_t* __restrict__ Z, int R, int C,
        const float* __restrict__ v, const float* __restrict__ state,
        float* __restrict__ zv, float* __restrict__ ns_norm) {
    __shared__ float s_dot[128];
    __shared__ float s_n2[128];
    int r = blockIdx.x;
    int tid = threadIdx.x;
    if (r >= R) return;
    float s1 = state[2];
    const precision_t* row = Z + (long)r * C;
    float dot = 0.0f, n2 = 0.0f;
    if (s1 >= 1.0f) {
        for (int c = tid; c < C; c += blockDim.x) {
            float z = to_float(row[c]);
            n2 += z * z;
        }
        s_n2[tid] = n2;
        __syncthreads();
        for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
            if (tid < s) s_n2[tid] += s_n2[tid + s];
            __syncthreads();
        }
        if (tid == 0) atomicAdd(ns_norm, s_n2[0]);
        return;
    }
    for (int c = tid; c < C; c += blockDim.x) {
        float z = to_float(row[c]);
        dot += z * v[c];
        n2 += z * z;
    }
    s_dot[tid] = dot;
    s_n2[tid] = n2;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) {
            s_dot[tid] += s_dot[tid + s];
            s_n2[tid] += s_n2[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        float zvr = s_dot[0];
        zv[r] = zvr;
        float row_capped_n2 = fmaxf(0.0f, s_n2[0] - (1.0f - s1 * s1) * zvr * zvr);
        atomicAdd(ns_norm, row_capped_n2);
    }
}

// Apply inner S and L2 normalize in one pass, with fast s1==1.0 skip.
__global__ void cap_r1_inner_apply_norm_kernel(
        precision_t* __restrict__ Z, int R, int C,
        const float* __restrict__ v, const float* __restrict__ zv,
        const float* __restrict__ state, const float* __restrict__ ns_norm,
        float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = R * C;
    if (idx < n) {
        float s1 = state[2];
        float inv_norm = 1.0f / fmaxf(sqrtf(*ns_norm), eps);
        if (s1 >= 1.0f) {
            Z[idx] = from_float(to_float(Z[idx]) * inv_norm);
            return;
        }
        int r = idx / C;
        int c = idx - r * C;
        float val = to_float(Z[idx]) - (1.0f - s1) * zv[r] * v[c];
        Z[idx] = from_float(val * inv_norm);
    }
}

// Project Q onto v; early-exit when s1 >= 1.0.
__global__ void cap_r1_outer_proj_kernel(
        const precision_t* __restrict__ Q, int R, int C,
        const float* __restrict__ v, const float* __restrict__ state,
        float* __restrict__ zv) {
    if (state[2] >= 1.0f) return;
    __shared__ float s_dot[128];
    int r = blockIdx.x;
    int tid = threadIdx.x;
    if (r >= R) return;
    const precision_t* row = Q + (long)r * C;
    float dot = 0.0f;
    for (int c = tid; c < C; c += blockDim.x) {
        dot += to_float(row[c]) * v[c];
    }
    s_dot[tid] = dot;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) s_dot[tid] += s_dot[tid + s];
        __syncthreads();
    }
    if (tid == 0) zv[r] = s_dot[0];
}

// Fused outer S + aspect scale store, with fast s1==1.0 skip.
__global__ void cap_r1_outer_store_kernel(
        precision_t* __restrict__ dst, const precision_t* __restrict__ Q,
        int R, int C, const float* __restrict__ v, const float* __restrict__ qv,
        const float* __restrict__ state, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int n = R * C;
    if (idx < n) {
        float s1 = state[2];
        if (s1 >= 1.0f) {
            dst[idx] = from_float(scale * to_float(Q[idx]));
            return;
        }
        int r = idx / C;
        int c = idx - r * C;
        float val = to_float(Q[idx]) - (1.0f - s1) * qv[r] * v[c];
        dst[idx] = from_float(scale * val);
    }
}
