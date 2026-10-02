// Scripted GPU regression capture across picking, stacking and basketball.
// Pass an output file; compile with ROBOT_HEADER to compare a saved implementation.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifndef ROBOT_HEADER
#define ROBOT_HEADER "../ocean/robot_arm/robot_arm_cuda.cuh"
#endif
#include ROBOT_HEADER
#define CUDA(c) do { auto err=(c); if(err!=cudaSuccess) { fprintf(stderr,"%s\n",cudaGetErrorString(err)); exit(2); } } while(0)
#ifndef ROBOT_ARM_CAPTURE_WORLDS
#define ROBOT_ARM_CAPTURE_WORLDS 32
#endif
constexpr int N=ROBOT_ARM_CAPTURE_WORLDS;
static_assert(N>0 && N<=1024, "Setup/capture use a single block");
__global__ void setup(Env* envs,int mode) {
    int i=threadIdx.x; if(i>=N) return;
    RaState* s=&envs[i].world.state;
    s->rng=i+1; s->stack_mode=mode==1; s->basketball_mode=mode==2;
    ra_reset(s); ra_rbrst(&envs[i].world.rigid,ra_topo(s));
    // Half of the worlds begin in contact with the pads; the others use reset RNG.
    if(i%2) {
        RaPose links[RA_LINKS]; RaVec3 ee;
        ra_fk(s->q,s->gripper_width,links,nullptr,nullptr,&ee);
        s->cube_position=ra_gctr(ee,links[RA_DOF].rotation);
        s->cube_rotation=links[RA_DOF].rotation;
        s->cube_velocity=ra_v3(0,0,0);
        s->gripper_width=0.060f;
    }
}
__global__ void act(float* actions,int step) {
    int i=threadIdx.x; if(i>=N) return;
    for(int j=0;j<7;++j) actions[8*i+j]=0.06f*sinf(0.025f*step+0.6f*j+0.1f*i);
    actions[8*i+7]=(step%120<80)?-1.0f:1.0f;
}
__global__ void snapshot(Env* envs,RaState* states,int* counts) {
    int i=threadIdx.x; if(i<N) {states[i]=envs[i].world.state;counts[i]=envs[i].world.rigid.manifold_count;}
}
int main(int argc,char** argv) {
    if(argc!=2 && argc!=3) return 2;
    int threads=argc==3?atoi(argv[2]):RA_CUDA_BLOCK_SIZE;
    if(threads!=32 && threads!=64 && threads!=128 && threads!=256) return 2;
    int blocks=(N+threads-1)/threads;
    FILE* f=fopen(argv[1],"wb"); if(!f) return 2;
    Env* e; float *a,*r,*t; obs_t* o; RaState* states; int* counts;
    CUDA(cudaMalloc(&e,N*sizeof(Env))); CUDA(cudaMalloc(&a,N*8*sizeof(float)));
    CUDA(cudaMalloc(&r,N*sizeof(float))); CUDA(cudaMalloc(&t,N*sizeof(float)));
    CUDA(cudaMalloc(&o,N*OBS_SIZE*sizeof(obs_t))); CUDA(cudaMalloc(&states,N*sizeof(RaState))); CUDA(cudaMalloc(&counts,N*sizeof(int)));
    std::vector<RaState> h(N); std::vector<float> io(N*(OBS_SIZE+2)); std::vector<int> hc(N);
    int contacts=0,nonfinite=0;
    for(int mode=0;mode<3;++mode) {
        CUDA(cudaMemset(e,0,N*sizeof(Env))); setup<<<1,N>>>(e,mode);
        for(int step=0;step<240;++step) {
            act<<<1,N>>>(a,step); ra_kbegin<<<blocks,threads>>>(e,0,N,a); ra_kphys<<<blocks,threads>>>(e,0,N); ra_kfin<<<blocks,threads>>>(e,0,N,o,r,t);
            snapshot<<<1,N>>>(e,states,counts); CUDA(cudaGetLastError());
            CUDA(cudaMemcpy(h.data(),states,N*sizeof(RaState),cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(hc.data(),counts,N*sizeof(int),cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(io.data(),o,N*OBS_SIZE*sizeof(float),cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(io.data()+N*OBS_SIZE,r,N*sizeof(float),cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(io.data()+N*(OBS_SIZE+1),t,N*sizeof(float),cudaMemcpyDeviceToHost));
            for(float v:io) nonfinite+=!std::isfinite(v);
            for(int v:hc) contacts+=v;
            fwrite(h.data(),sizeof(RaState),N,f); fwrite(io.data(),sizeof(float),io.size(),f); fwrite(hc.data(),sizeof(int),N,f);
        }
    }
    fclose(f); printf("world_transitions=%d contact_manifolds=%d nonfinite=%d\n",N*3*240,contacts,nonfinite);
    CUDA(cudaFree(counts)); CUDA(cudaFree(states)); CUDA(cudaFree(o));
    CUDA(cudaFree(t)); CUDA(cudaFree(r)); CUDA(cudaFree(a)); CUDA(cudaFree(e));
    return nonfinite || contacts==0 ? 1 : 0;
}
