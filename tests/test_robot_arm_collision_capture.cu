// Captures focused compound contacts and swept queries for old/new comparisons.
// Set ROBOT_HEADER to a saved header and ROBOT_ARM_STAGE1 for the old API.
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifndef ROBOT_HEADER
#define ROBOT_HEADER "../ocean/robot_arm/robot_arm_cuda.cuh"
#endif
#include ROBOT_HEADER
#define CUDA(call) do { auto err=(call); if(err!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(err)); std::exit(2); } } while(0)
constexpr int N=128;
struct Capture {
    int contacts[2]; uint32_t masks[2]; PlImpulseManifold manifolds[2];
    RaConvexSweep shape_sweep,ring_sweep; RaConvexContact ring_contact;
};
__global__ void capture(Env* envs,Capture* out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=N) return;
    auto* world=&envs[i].world; auto* s=&world->state;
    s->rng=i+1; s->stack_mode=i%3==1; s->basketball_mode=i%3==2;
    ra_reset(s); ra_rbrst(&world->rigid,ra_topo(s));
    for(int j=0;j<7;++j) s->q[j]+=0.07f*sinf(0.23f*i+j);
    s->gripper_width=0.01f+0.07f*(i%8)/7.0f;
    ra_fk(s->q,s->gripper_width,world->staged.links,world->staged.origins,world->staged.axes,&s->end_effector);
    RaQuat hand=world->staged.links[RA_DOF].rotation;
    s->cube_position=ra_add(ra_gctr(s->end_effector,hand),
        ra_rotate(hand,ra_v3(0.014f*sinf(i*0.37f),0.01f*sinf(i*0.71f),0.012f*cosf(i*0.19f))));
    s->cube_rotation=ra_qmul(hand,ra_qaxis(ra_v3(0,1,0),0.15f*sinf(i*0.53f)));
    s->cube_velocity=ra_v3(0.03f*sinf(i),-0.05f,0.02f*cosf(i));
    ra_bodies(world);
    for(int side=0;side<2;++side) {
        int before=world->rigid.manifold_count;
#ifdef ROBOT_ARM_STAGE1
        out[i].contacts[side]=s->basketball_mode?ra_bpad(world,side):ra_padc(world,side,RA_CUDA_BODY_CUBE);
#else
        out[i].contacts[side]=ra_pad_contact(world,side,RA_CUDA_BODY_CUBE);
#endif
        out[i].masks[side]=world->rigid.compound_pad_component_mask[side];
        if(out[i].contacts[side]) out[i].manifolds[side]=world->rigid.manifolds[before];
    }
    RaVec3 v=ra_v3(0.3f,-2.0f-0.05f*i,0.1f);
    RaVec3 angular=ra_v3(0.1f*i,0.03f*i,0.02f*i),zero=ra_v3(0,0,0);
#ifdef ROBOT_ARM_STAGE1
    out[i].shape_sweep=ra_boxccd(&world->rigid.shapes[RA_CUDA_BODY_CUBE],&world->rigid.shapes[RA_CUDA_BODY_TABLE],
        v,angular,zero,zero,0.2f,0.0005f);
#else
    out[i].shape_sweep=RaSweep::shape(&world->rigid.shapes[RA_CUDA_BODY_CUBE],&world->rigid.shapes[RA_CUDA_BODY_TABLE],
        v,angular,zero,zero,0.2f,0.0005f);
#endif
    RaRigidBody ball={};
    ball.pose.position=ra_add(ra_hoop(),ra_v3(RA_RIM_MAJOR_RADIUS*cosf(0.3f*i),
        -0.06f+0.002f*(i%64),RA_RIM_MAJOR_RADIUS*sinf(0.3f*i)));
    if(i%16==0) ball.pose.position=ra_hoop();
    ball.linear_velocity=ra_v3(0.2f*sinf(i),-0.8f,0.2f*cosf(i));
    out[i].ring_contact=ra_rimq(ball.pose.position,0.0005f);
    out[i].ring_sweep=ra_rimccd(&ball,0.2f);
}
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    Env* envs; Capture* device;
    CUDA(cudaMalloc(&envs,N*sizeof(Env))); CUDA(cudaMemset(envs,0,N*sizeof(Env)));
    CUDA(cudaMalloc(&device,N*sizeof(Capture))); CUDA(cudaMemset(device,0,N*sizeof(Capture)));
    capture<<<1,N>>>(envs,device); CUDA(cudaGetLastError());
    std::vector<Capture> h(N); CUDA(cudaMemcpy(h.data(),device,N*sizeof(Capture),cudaMemcpyDeviceToHost));
    int contacts=0,shape_hits=0,ring_hits=0;
    for(const auto& c:h) {contacts+=c.contacts[0]+c.contacts[1];shape_hits+=c.shape_sweep.hit;ring_hits+=c.ring_sweep.hit;}
    FILE* f=std::fopen(argv[1],"wb"); if(!f) return 2;
    bool written=std::fwrite(h.data(),sizeof(Capture),N,f)==N; std::fclose(f);
    CUDA(cudaFree(device)); CUDA(cudaFree(envs));
    std::printf("fixtures=%d compound_manifolds=%d shape_hits=%d ring_hits=%d\n",N,contacts,shape_hits,ring_hits);
    return !written || contacts==0 || shape_hits==0 || ring_hits==0 ? 1 : 0;
}
