// Independent analytical checks, compiled as C++ or CUDA.
#include <cassert>
#include <cmath>
#include <cstdio>
#ifndef __CUDACC__
#define __host__
#define __device__
#define __forceinline__ inline
static float host_rsqrt(float x) { return 1.0f/std::sqrt(x); }
#define rsqrtf host_rsqrt
#else
#include <cuda_runtime.h>
#endif
#include "../src/puffysics/serial_dynamics.cuh"
using D = PfSerialDynamicsT<2>;
struct Chain {
    __device__ static float mass(int b) { return b == 0 ? 2 : 3; }
    __device__ static PfVec3 com(int) { return pf_v3(0,0,0); }
    __device__ static int last(int b) { return b == 2 ? -1 : b; }
    __device__ static D::Tensor inertia(int b) { return {0.2f,0.2f,b==0?0.3f:0.4f,0,0,0}; }
};
using S = PfSerialDynamicsT<1>;
struct TensorBody {
    __device__ static float mass(int) { return 1; }
    __device__ static PfVec3 com(int) { return pf_v3(0,0,0); }
    __device__ static int last(int) { return 0; }
    __device__ static S::Tensor inertia(int) { return {0.2f,0.4f,0.6f,0.1f,0,0}; }
};
__device__ static void near(float a,float b) { assert(fabsf(a-b)<3e-5f); }
__device__ static void check() {
    PfPose bodies[3] = {{pf_v3(.5f,0,0),pf_quat_identity()},
        {pf_v3(2,0,0),pf_quat_identity()}, {pf_v3(10,0,0),pf_quat_identity()}};
    PfVec3 origins[2]={pf_v3(0,0,0),pf_v3(1,0,0)};
    PfVec3 axes[2]={pf_v3(0,0,1),pf_v3(0,0,1)};
    float armature[2]={.1f,.2f}, m[2][2], l[2][2], g[2];
    D::mass<Chain>(bodies,origins,axes,3,armature,m);
    near(m[0][0],13.3f); near(m[0][1],6.4f); near(m[1][0],6.4f); near(m[1][1],3.6f);
    D::gravity<Chain>(bodies,origins,axes,3,pf_v3(0,-9.81f,0),g);
    near(g[0],-7*9.81f); near(g[1],-3*9.81f);
    D::factor(m,l);
    float result[2], rhs[2]={1,-2}; D::solve(l,rhs,result);
    near(m[0][0]*result[0]+m[0][1]*result[1],rhs[0]);
    near(m[1][0]*result[0]+m[1][1]*result[1],rhs[1]);
    float j[2],r[2],im;
    D::response(l,origins,axes,1,pf_v3(2,0,0),pf_v3(0,1,0),pf_v3(0,0,0),j,r,&im);
    near(j[0],2); near(j[1],1);
    float det=13.3f*3.6f-6.4f*6.4f;
    near(r[0],(.8f)/det); near(r[1],.5f/det); near(im,2.1f/det);
    D::response(l,origins,axes,0,pf_v3(2,0,0),pf_v3(0,0,0),pf_v3(0,0,1),j,r,&im);
    near(j[0],1); near(j[1],0); near(im,3.6f/det);
    D::response(l,origins,axes,-1,pf_v3(2,0,0),pf_v3(0,1,0),pf_v3(0,0,1),j,r,&im);
    near(im,0); near(r[0],0); near(r[1],0);
    float qd[2]={2,-1};
    near(D::point_velocity(qd,origins,axes,1,pf_v3(2,0,0)).y,3);
    near(D::angular_velocity(qd,axes,1).z,1);
    PfVec3 diagonal_axis=pf_v3(sqrtf(.5f),sqrtf(.5f),0);
    PfPose pose={pf_v3(0,0,0),pf_quat_identity()}; float a[1]={0}, tensor_mass[1][1];
    S::mass<TensorBody>(&pose,origins,&diagonal_axis,1,a,tensor_mass);
    near(tensor_mass[0][0],.4f); // includes the off-diagonal xy inertia
    pose.rotation=pf_quat_from_axis_angle(pf_v3(0,0,1),1.57079632679f);
    S::mass<TensorBody>(&pose,origins,&diagonal_axis,1,a,tensor_mass);
    near(tensor_mass[0][0],.2f); // rotating the full tensor changes effective inertia
    float singular[1][1]={{0}}, regularized[1][1];
    S::factor(singular,regularized,1e-6f); near(regularized[0][0],.001f);
}
#ifdef __CUDACC__
__global__ void test_kernel() { check(); }
#endif
int main() {
#ifdef __CUDACC__
    test_kernel<<<1,1>>>();
    auto err=cudaDeviceSynchronize();
    if(err!=cudaSuccess) { fprintf(stderr,"%s\n",cudaGetErrorString(err)); return 1; }
#else
    check();
#endif
    puts("serial dynamics analytical checks passed");
}
