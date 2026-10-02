#include <dlfcn.h>
#include "../ocean/robot_arm/robot_arm.cu"

// Exercise the real renderer without a policy or training loop. Run one mode
// per process; the optional screenshot uses PUFFER_ROBOT_ARM_SCREENSHOT.
int main(int argc, char** argv) {
    int mode = argc > 1 ? atoi(argv[1]) : 0;
    assert(mode >= 0 && mode <= 2);
    g_ra_stack = mode == 1;
    g_ra_basketball = mode == 2;
    g_ra_render_host.model_glb = g_ra_model_glb;
    g_ra_render_host.camera_distance = mode == 2 ? 2.35f : 1.55f;
    g_ra_render_host.camera_yaw = 0.78f;
    g_ra_render_host.camera_pitch = 0.48f;
    Env* host = (Env*)calloc(1, sizeof(Env));
    ra_fill(host, 17);
    assert(cudaMalloc(&g_gpu.envs, sizeof(Env)) == cudaSuccess);
    assert(cudaMemcpy(g_gpu.envs, host, sizeof(Env), cudaMemcpyHostToDevice) == cudaSuccess);
    void* gl = dlopen("libGL.so.1", RTLD_LAZY);
    assert(gl != NULL);
    unsigned int (*gl_error)(void) = (unsigned int (*)(void))dlsym(gl, "glGetError");
    assert(gl_error != NULL);
    for (int frame = 0; frame < 4; ++frame) {
        if (frame == 2) {
            host->world.state.q[0] += 0.35f;
            host->world.state.cube_position.y += 0.08f;
            assert(
                cudaMemcpy(g_gpu.envs, host, sizeof(Env), cudaMemcpyHostToDevice) == cudaSuccess);
        }
        puf_render(NULL);
        assert(gl_error() == 0);
    }
    puf_close(NULL);
    free(host);
    dlclose(gl);
    printf("PASS robot_arm rendering mode=%d\n", mode);
}
