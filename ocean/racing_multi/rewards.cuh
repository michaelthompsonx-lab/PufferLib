#pragma once

static constexpr float RACING_PROGRESS_METRES_PER_REWARD = 1000.0f;
static constexpr float RACING_DNF_REWARD = -7.0f;
static constexpr float RACING_POSITION_REWARD = 3.0f;
static constexpr float RACING_LAP_REWARD = 3.5f;
static constexpr float RACING_LAP_SPEED_REWARD = 2.5f;
// Leave headroom for progress, position changes and contact costs on terminal steps.
static constexpr float RACING_REWARD_SCALE = 0.1f;

__host__ __device__ static float racing_progress_reward(const RacingTask &task,
    float previous_furthest, float limit) {
    if (task.offroad || task.route_jump) return 0;
    // Furthest includes off-road travel: returning cannot claim it again.
    return racing_clamp((fminf(task.furthest, limit) - fminf(previous_furthest, limit)) /
        RACING_PROGRESS_METRES_PER_REWARD, 0, 0.02f);
}

__host__ __device__ static float racing_lap_bonus(float finish_seconds, float limit) {
    return RACING_LAP_REWARD + RACING_LAP_SPEED_REWARD
        * racing_clamp(1 - finish_seconds / limit, 0, 1);
}

__host__ __device__ static float racing_reward_for_ppo(float reward) {
    return reward * RACING_REWARD_SCALE;
}
