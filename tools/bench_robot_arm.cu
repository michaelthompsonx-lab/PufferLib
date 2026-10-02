#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include "../ocean/robot_arm/robot_arm_cuda.cuh"

// Existing production begin/physics/finish kernels, without policy inference,
// rendering or trainer copies. This is a baseline, not a native-path speedup.
#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(2); } } while(0)
__global__ static void initialize(Env* envs,int n,int mode,float* actions) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    Env& e=envs[i]; e.num_agents=1; e.rng=i+1; e.world.state.rng=i+1;
    e.world.state.stack_mode=mode==1; e.world.state.basketball_mode=mode==2;
    ra_reset(&e.world.state); ra_rbrst(&e.world.rigid,ra_topo(&e.world.state));
    for(int a=0;a<8;++a) actions[i*8+a]=a==7?1:0;
}
struct Result { int nonfinite, terminal; float reward,y; };
__global__ static void inspect(Env* envs,int n,const obs_t* obs,const float* rewards,
        const float* terminals,Result* out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    Result r={}; r.reward=rewards[i]; r.y=envs[i].world.state.cube_position.y;
    for(int k=0;k<OBS_SIZE;++k) r.nonfinite+=!isfinite((float)obs[i*OBS_SIZE+k]);
    for(int k=0;k<7;++k) r.nonfinite+=!isfinite(envs[i].world.state.q[k]) || !isfinite(envs[i].world.state.qd[k]);
    r.nonfinite+=!isfinite(r.reward) || !isfinite(r.y);
    r.terminal=terminals[i]!=0; out[i]=r;
}
int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IOLBF,0);
    int n=argc>1?atoi(argv[1]):4096, calls=argc>2?atoi(argv[2]):100;
    int threads=argc>3?atoi(argv[3]):RA_CUDA_BLOCK_SIZE;
    if(n<=0 || calls<=0 || (threads!=32 && threads!=64 && threads!=128 && threads!=256)) return 2;
    cudaDeviceProp p; CUDA(cudaGetDeviceProperties(&p,0));
    printf("GPU=%s worlds=%d calls=%d control_dt=%g substeps=%d threads=%d workload=production_hold_open\n",
        p.name,n,calls,RA_CONTROL_DT,RA_SUBSTEPS,threads);
    cudaFuncAttributes attributes; CUDA(cudaFuncGetAttributes(&attributes,ra_kphys));
    printf("physics_registers=%d local_bytes=%zu shared_bytes=%zu gpu_sms=%d\n",
        attributes.numRegs,attributes.localSizeBytes,attributes.sharedSizeBytes,p.multiProcessorCount);
    Env* envs; float *actions,*rewards,*terminals; obs_t* obs; Result* out;
    CUDA(cudaMalloc((void**)&envs,(size_t)n*sizeof(Env)));
    CUDA(cudaMalloc((void**)&actions,(size_t)n*8*sizeof(float)));
    CUDA(cudaMalloc((void**)&rewards,n*sizeof(float))); CUDA(cudaMalloc((void**)&terminals,n*sizeof(float)));
    CUDA(cudaMalloc((void**)&obs,(size_t)n*OBS_SIZE*sizeof(obs_t))); CUDA(cudaMalloc((void**)&out,n*sizeof(Result)));
    size_t bytes=(size_t)n*(sizeof(Env)+(8+2)*sizeof(float)+OBS_SIZE*sizeof(obs_t));
    printf("allocated_MiB=%.2f bytes_per_env=%zu observation_bytes=%zu\n",bytes/1048576.0,sizeof(Env),sizeof(obs_t));
    cudaStream_t stream; CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    cudaEvent_t start,stop; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop));
    int blocks=(n+threads-1)/threads;
    auto step=[&]() {
        ra_kbegin<<<blocks,threads,0,stream>>>(envs,0,n,actions);
        ra_kphys<<<blocks,threads,0,stream>>>(envs,0,n);
        ra_kfin<<<blocks,threads,0,stream>>>(envs,0,n,obs,rewards,terminals);
        CUDA(cudaGetLastError());
    };
    for(int mode=0;mode<3;++mode) {
        std::vector<float> times;
        for(int repeat=0;repeat<5;++repeat) {
            CUDA(cudaMemsetAsync(envs,0,(size_t)n*sizeof(Env),stream));
            initialize<<<blocks,threads,0,stream>>>(envs,n,mode,actions); CUDA(cudaGetLastError());
            for(int i=0;i<20;++i) step();
            CUDA(cudaEventRecord(start,stream)); for(int i=0;i<calls;++i) step();
            CUDA(cudaEventRecord(stop,stream)); CUDA(cudaEventSynchronize(stop));
            float ms; CUDA(cudaEventElapsedTime(&ms,start,stop)); times.push_back(ms);
            inspect<<<blocks,threads,0,stream>>>(envs,n,obs,rewards,terminals,out);
            CUDA(cudaGetLastError()); CUDA(cudaStreamSynchronize(stream));
            std::vector<Result> results(n); CUDA(cudaMemcpy(results.data(),out,n*sizeof(Result),cudaMemcpyDeviceToHost));
            int failures=0; float min_y=1e30f,max_y=-1e30f;
            for(Result r:results) { failures+=r.nonfinite; min_y=fminf(min_y,r.y); max_y=fmaxf(max_y,r.y); }
            printf("mode=%d repeat=%d ms=%.3f nonfinite=%d object_y=[%g,%g]\n",mode,repeat,ms,failures,min_y,max_y);
            if(failures) return 1;
        }
        std::sort(times.begin(),times.end());
        printf("mode=%d median_ms=%.3f control_steps_per_s=%.0f substeps_per_s=%.0f batch_control_ms=%.3f\n",
            mode,times[2],n*calls*1000.0/times[2],n*calls*8000.0/times[2],times[2]/calls);
    }
    CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(stop)); CUDA(cudaStreamDestroy(stream));
    CUDA(cudaFree(envs)); CUDA(cudaFree(actions)); CUDA(cudaFree(rewards)); CUDA(cudaFree(terminals));
    CUDA(cudaFree(obs)); CUDA(cudaFree(out));
}
