#pragma once

// Captured before the action: a transition ending in DNF/finish remains trainable.
__global__ static void racing_capture_active(const Env *envs, precision_t *mask, int count) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < count) mask[i] = from_float(envs[i].task.done ? 0.0f : 1.0f);
}

// One block per recurrent minibatch. The last observation normally bootstraps GAE;
// a terminal final action has a known target and is retained using the env's tail reward.
__global__ static void racing_count_active(precision_t *mask, const float *tail_done,
    float *counts, int rows_per_batch, int horizon) {
    __shared__ int sums[256];
    int tid = threadIdx.x, count = 0;
    int first_row = blockIdx.x*rows_per_batch;
    for (int k = tid; k < rows_per_batch*horizon; k += blockDim.x) {
        int row = first_row+k/horizon, t = k%horizon;
        int index = row*horizon+t;
        if (t == horizon-1 && tail_done[row] == 0) mask[index] = from_float(0.0f);
        count += to_float(mask[index]) != 0;
    }
    sums[tid] = count;
    __syncthreads();
    for (int width=128; width; width/=2) {
        if (tid < width) sums[tid] += sums[tid+width];
        __syncthreads();
    }
    if (!tid) counts[blockIdx.x] = sums[0];
}
