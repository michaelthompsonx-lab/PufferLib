// Compare boolean task clearance and geometry before/after the extraction.
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifndef ROBOT_HEADER
#define ROBOT_HEADER "../ocean/robot_arm/robot_arm_cuda.cuh"
#endif
#include ROBOT_HEADER
struct Capture { int hit[2]; RaPose pad[5]; RaVec3 extent[5]; RaPose shell[7]; RaVec3 shell_extent[7]; };
__global__ void capture(Capture* output) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=4096) return;
    RaPose finger={ra_v3(.3f,-.2f,.1f),ra_qmul(ra_qaxis(ra_v3(1,0,0),.013f*(i%257)),
        ra_qmul(ra_qaxis(ra_v3(0,1,0),.017f*(i%131)),ra_qaxis(ra_v3(0,0,1),.011f*(i%197))))};
    RaQuat rotation=ra_qmul(finger.rotation,ra_qaxis(ra_v3(1,0,0),i%16==0?1e-7f:.017f*(i%89)));
    RaVec3 offset=ra_v3(.07f*sinf(i*.19f),.065f*cosf(i*.11f),.0445f+.055f*sinf(i*.23f));
    if(i%16==0) offset=ra_v3(0,-.0335f,.0445f);
    RaVec3 position=ra_add(finger.position,ra_rotate(finger.rotation,offset));
    Capture c={};
    for(int m=0;m<2;++m) {
        float margin=m?.002f:0;
#ifdef ROBOT_ARM_STAGE3
        RaConvexContact ignored;
        c.hit[m]=ra_padhit(position,rotation,finger,margin,&ignored);
#else
        c.hit[m]=ra_pad_clearance(position,rotation,finger,margin);
#endif
    }
    for(int p=0;p<5;++p) { auto b=ra_padsh(finger,p);c.pad[p]=b.pose;c.extent[p]=b.half_extents; }
    RaGripperCollisionFrame f={finger,finger,finger};
    for(int b=0;b<7;++b) { auto box=ra_gripb(&f,b);c.shell[b]=box.pose;c.shell_extent[b]=box.half_extent; }
    output[i]=c;
}
int main(int argc,char** argv) {
    if(argc!=2) return 2;
    Capture* d; auto err=cudaMalloc(&d,4096*sizeof(Capture)); if(err!=cudaSuccess) return 2;
    capture<<<32,128>>>(d); err=cudaDeviceSynchronize();
    if(err!=cudaSuccess) { fprintf(stderr,"%s\n",cudaGetErrorString(err)); return 2; }
    std::vector<Capture> h(4096); err=cudaMemcpy(h.data(),d,h.size()*sizeof(Capture),cudaMemcpyDeviceToHost);
    if(err!=cudaSuccess) return 2;
    FILE* f=fopen(argv[1],"wb"); if(!f) return 2;
    size_t n=fwrite(h.data(),sizeof(Capture),h.size(),f); fclose(f); cudaFree(d);
    int hits=0;for(auto c:h) hits+=c.hit[0]+c.hit[1];
    printf("fixtures=%zu clearance_hits=%d\n",h.size(),hits);
    return n==h.size()?0:2;
}
