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
// step energy.
//
// 100% BIT-FOR-BIT DETERMINISTIC & CUDA-GRAPH COMPATIBLE:
//   - Zero atomicAdd in shared or global memory (per-warp disjoint SMEM ->
//     fixed-order warp sum -> per-block partials -> fixed-order block sum, and
//     deterministic row-norm partial reduction).
//   - Zero host syncs, zero dense d x d buffers, supports up to d <= 4096.

constexpr int MAX_CAP_LAYERS = 16;
constexpr int CAP_MAX_BLOCKS = 64;
constexpr int MAX_CAP_SMEM_BYTES = 48 * 1024;

struct CapLayer {
    bool enabled;
    int kind;        // 0=encoder, 1=mingru, 2=decoder
    int kind_idx;    // layer index within kind (for mingru)
    int d;           // in_dim (columns C)
    int out_dim;     // out_dim (rows R)
    int sketch_warps;// warps/block for disjoint SMEM sketch (4, 2, or 1)
    long param_off;  // element offset of weight in flat param/grad buffer
    float ema_gamma; // EMA decay (default 0.9)
    float cap_c;     // threshold ratio c (default 1.0)
    float cap_p;     // cap power p (default 0.5)

    // Device buffers (all O(d) or O(out_dim), zero atomicAdd)
    float* v1;       // [d] unit top eigenvector
    float* part;     // [CAP_MAX_BLOCKS * (2*d + 2)] per-block deterministic partials
    float* scratch;  // [d] scratch vector for finalize (mean or u)
    float* state;    // [4] {lam1, tr_ema, s1, steps}
    float* zv;       // [out_dim] scratch row projections Z*v or Q*v
    float* row_n2;   // [out_dim] per-row capped Frobenius norm^2 (deterministic reduce)
};

struct CapManager {
    int num_layers;
    CapLayer layers[MAX_CAP_LAYERS];
};

inline void cap_layer_init(CapLayer* c, int kind, int kind_idx,
        int d, int out_dim, long param_off,
        float ema_gamma, float cap_c, float cap_p) {
    memset(c, 0, sizeof(*c));
    if (d <= 0 || out_dim <= 0) {
        return;
    }
    // Choose highest warp count whose disjoint SMEM (1 + 2*nwarps)*d*4 fits in 48KB:
    //   4 warps: d <= 1365, 2 warps: d <= 2457, 1 warp: d <= 4096.
    int warps = 0;
    if ((1 + 2 * 4) * d * (int)sizeof(float) <= MAX_CAP_SMEM_BYTES) {
        warps = 4;
    } else if ((1 + 2 * 2) * d * (int)sizeof(float) <= MAX_CAP_SMEM_BYTES) {
        warps = 2;
    } else if ((1 + 2 * 1) * d * (int)sizeof(float) <= MAX_CAP_SMEM_BYTES) {
        warps = 1;
    } else {
        return;
    }

    c->enabled = true;
    c->kind = kind;
    c->kind_idx = kind_idx;
    c->d = d;
    c->out_dim = out_dim;
    c->sketch_warps = warps;
    c->param_off = param_off;
    c->ema_gamma = ema_gamma;
    c->cap_c = cap_c;
    c->cap_p = cap_p;

    int part_stride = 2 * d + 2;
    cudaMalloc((void**)&c->v1, d * sizeof(float));
    cudaMalloc((void**)&c->part, (size_t)CAP_MAX_BLOCKS * part_stride * sizeof(float));
    cudaMalloc((void**)&c->scratch, d * sizeof(float));
    cudaMalloc((void**)&c->state, 4 * sizeof(float));
    cudaMalloc((void**)&c->zv, out_dim * sizeof(float));
    cudaMalloc((void**)&c->row_n2, out_dim * sizeof(float));

    cudaMemset(c->v1, 0, d * sizeof(float));
    cudaMemset(c->part, 0, (size_t)CAP_MAX_BLOCKS * part_stride * sizeof(float));
    cudaMemset(c->scratch, 0, d * sizeof(float));
    cudaMemset(c->zv, 0, out_dim * sizeof(float));
    cudaMemset(c->row_n2, 0, out_dim * sizeof(float));
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

// 100% deterministic warp-disjoint accumulation (ZERO atomicAdd) of sum_x,
// sum_x*(x.v), sum ||x||^2, and sum (x.v)^2 into per-block partial slots.
__global__ void cap_sketch_accum_kernel(
        const precision_t* __restrict__ X, int B, int d,
        const float* __restrict__ v1, float* __restrict__ part) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int warp = tid >> 5;
    int lane = tid & 31;
    int nwarps = blockDim.x >> 5;

    float* s_v    = smem;                          // [d]
    float* s_mean = smem + d;                      // [nwarps * d]
    float* s_gv   = smem + d + nwarps * d;         // [nwarps * d]
    __shared__ float s_sc[4][2];

    for (int j = tid; j < d; j += blockDim.x) {
        s_v[j] = v1[j];
    }
    for (int j = lane; j < d; j += 32) {
        s_mean[warp * d + j] = 0.0f;
        s_gv[warp * d + j] = 0.0f;
    }
    float w_tr = 0.0f;
    float w_lam = 0.0f;
    __syncthreads();

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
            w_tr += tr;
            w_lam += y * y;
        }
        // Each lane owns disjoint columns j = lane, lane+32, ... inside its own warp slice -> zero atomics.
        for (int j = lane; j < d; j += 32) {
            float xj = to_float(row[j]);
            s_mean[warp * d + j] += xj;
            s_gv[warp * d + j] += xj * y;
        }
    }
    if (lane == 0) {
        s_sc[warp][0] = w_tr;
        s_sc[warp][1] = w_lam;
    }
    __syncthreads();

    // Fixed-order sum across warps 0..nwarps-1 into this block's partial slot.
    int stride = 2 * d + 2;
    float* out = part + (long)blockIdx.x * stride;
    for (int j = tid; j < d; j += blockDim.x) {
        float m = 0.0f, g = 0.0f;
        for (int w = 0; w < nwarps; w++) {
            m += s_mean[w * d + j];
            g += s_gv[w * d + j];
        }
        out[j] = m;
        out[d + j] = g;
    }
    if (tid < 2) {
        float sc = 0.0f;
        for (int w = 0; w < nwarps; w++) {
            sc += s_sc[w][tid];
        }
        out[2 * d + tid] = sc;
    }
}

// Single-block deterministic finalize: sums blocks 0..nblk-1 in fixed order
// and updates Rayleigh-EMA (v1, lam1, tr_ema, s1).
__global__ void cap_sketch_finalize_kernel(
        int B, int d, int nblk, float ema_gamma, float cap_c, float cap_p,
        float* __restrict__ v1, const float* __restrict__ part,
        float* __restrict__ scratch, float* __restrict__ state) {
    __shared__ float s_buf[256];
    __shared__ float s_tot_sc[2];
    int tid = threadIdx.x;
    int stride = 2 * d + 2;
    float invB = 1.0f / (float)B;

    if (tid < 2) {
        float sc = 0.0f;
        for (int b = 0; b < nblk; b++) {
            sc += part[(long)b * stride + 2 * d + tid];
        }
        s_tot_sc[tid] = sc;
    }
    __syncthreads();

    float prev_lam1 = state[0];
    float prev_tr   = state[1];
    float steps     = state[3];
    float b_tr      = s_tot_sc[0] * invB;
    float b_lam1    = s_tot_sc[1] * invB;

    if (steps < 0.5f) {
        float mn2 = 0.0f;
        for (int j = tid; j < d; j += blockDim.x) {
            float m = 0.0f;
            for (int b = 0; b < nblk; b++) {
                m += part[(long)b * stride + j];
            }
            m *= invB;
            scratch[j] = m;
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
            v1[j] = (mn > 1e-12f) ? (scratch[j] / mn) : ((j == 0) ? 1.0f : 0.0f);
        }
        if (tid == 0) {
            state[0] = b_lam1;
            state[1] = b_tr;
            state[2] = 1.0f;
            state[3] = 1.0f;
        }
        return;
    }

    float g = (steps < 1.5f) ? 0.0f : ema_gamma;
    float lam1 = g * prev_lam1 + (1.0f - g) * b_lam1;
    float tr_ema = g * prev_tr + (1.0f - g) * b_tr;

    float u_norm2 = 0.0f;
    for (int j = tid; j < d; j += blockDim.x) {
        float gv = 0.0f;
        for (int b = 0; b < nblk; b++) {
            gv += part[(long)b * stride + d + j];
        }
        gv *= invB;
        float uj = g * prev_lam1 * v1[j] + (1.0f - g) * gv;
        scratch[j] = uj;
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
        if (un > 1e-12f) v1[j] = scratch[j] / un;
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
    }
}

inline void cap_sketch_update(CapLayer* c, Prec saved_input, cudaStream_t stream) {
    if (!c->enabled) return;
    int d = c->d;
    long tot = numel(saved_input.shape);
    if (d <= 0 || tot <= 0 || (tot % d) != 0) return;
    int B = (int)(tot / d);
    int nwarps = c->sketch_warps;
    int blocks = min((B + nwarps - 1) / nwarps, CAP_MAX_BLOCKS);
    size_t smem = (size_t)((1 + 2 * nwarps) * d) * sizeof(float);
    cap_sketch_accum_kernel<<<blocks, nwarps * 32, smem, stream>>>(
        saved_input.data, B, d, c->v1, c->part);
    cap_sketch_finalize_kernel<<<1, 256, 0, stream>>>(
        B, d, blocks, c->ema_gamma, c->cap_c, c->cap_p,
        c->v1, c->part, c->scratch, c->state);
}

// Deterministic per-row rank-1 inner S projection + per-row capped Frobenius norm^2
// (writes to disjoint row_n2[r], zero atomicAdd).
__global__ void cap_r1_inner_proj_norm_kernel(
        const precision_t* __restrict__ Z, int R, int C,
        const float* __restrict__ v, const float* __restrict__ state,
        float* __restrict__ zv, float* __restrict__ row_n2) {
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
        if (tid == 0) row_n2[r] = s_n2[0];
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
        row_n2[r] = fmaxf(0.0f, s_n2[0] - (1.0f - s1 * s1) * zvr * zvr);
    }
}

// Single-block deterministic sum reduction over row_n2[0..R-1] -> *ns_norm.
__global__ void cap_r1_norm_reduce_kernel(
        const float* __restrict__ row_n2, int R, float* __restrict__ ns_norm) {
    __shared__ float s_buf[256];
    int tid = threadIdx.x;
    float sum = 0.0f;
    for (int r = tid; r < R; r += blockDim.x) {
        sum += row_n2[r];
    }
    s_buf[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s) s_buf[tid] += s_buf[tid + s];
        __syncthreads();
    }
    if (tid == 0) *ns_norm = s_buf[0];
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

// Deterministic inner S + Frobenius L2 normalization helper (zero atomicAdd).
inline void cap_r1_inner_apply(
        CapLayer* cl, precision_t* Z, int R, int C,
        float* ns_norm, float eps, int block_size, cudaStream_t stream) {
    int ne = R * C;
    int grid = (ne + block_size - 1) / block_size;
    cap_r1_inner_proj_norm_kernel<<<R, 128, 0, stream>>>(
        Z, R, C, cl->v1, cl->state, cl->zv, cl->row_n2);
    cap_r1_norm_reduce_kernel<<<1, 256, 0, stream>>>(
        cl->row_n2, R, ns_norm);
    cap_r1_inner_apply_norm_kernel<<<grid, block_size, 0, stream>>>(
        Z, R, C, cl->v1, cl->zv, cl->state, ns_norm, eps);
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
