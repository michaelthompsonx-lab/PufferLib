// Independent CPU/CUDA checks for explicit updates and box clearance.
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
#include "../src/puffysics/explicit_dynamics.cuh"
#include "../src/puffysics/box_clearance.cuh"
using E = PfExplicitDynamicsT<>;
using C = PfBoxClearanceT<>;
struct Tensor { float xx,yy,zz,xy,xz,yz; };
__device__ static void near(float a,float b) { assert(fabsf(a-b)<1e-5f); }
__device__ static void check() {
    near(E::servo(2,1,.5f,10,2,4),4);
    near(E::servo(0,1,.5f,10,2,4),-4);
    near(E::servo(1.2f,1,.5f,10,2,4),1);
    near(E::velocity(1,4,.5f,2),2);
    float v=-2;
    near(E::coordinate(.1f,v,.1f,0,1,.15f),0); near(v,0);
    v=2; near(E::coordinate(.9f,v,.1f,0,1,.15f),1); near(v,0);
    v=2; near(E::coordinate(.2f,v,.1f,0,1,.15f),.4f); near(v,2);
    // Exactly at the boundary is distinct for joints and clamped slides.
    v=-1; near(E::coordinate(.1f,v,.1f,0,1,.15f),0); near(v,-1);
    v=-1; near(E::slide(.1f,v,.1f,0,1),0); near(v,0);
    v=1; near(E::slide(.9f,v,.1f,0,1),1); near(v,0);
    auto soft=E::compliance(100,1,1,.1f,1e-8f);
    near(soft.cfm,1.0f/3); near(soft.erp,10.0f/3);
    auto box=E::box_inertia<Tensor>(3,pf_v3(1,2,3));
    near(box.xx,13); near(box.yy,10); near(box.zz,5);
    auto sphere=E::sphere_inertia<Tensor>(5,2); near(sphere.xx,8);
    Tensor t={2,3,4,.2f,.1f,.3f};
    PfVec3 input=pf_v3(1,2,3);
    auto inverse=E::inverse_inertia(t,pf_quat_identity(),input,1e-18f);
    near(t.xx*inverse.x+t.xy*inverse.y+t.xz*inverse.z,1);
    near(t.xy*inverse.x+t.yy*inverse.y+t.yz*inverse.z,2);
    near(t.xz*inverse.x+t.yz*inverse.y+t.zz*inverse.z,3);
    auto rotation=pf_quat_from_axis_angle(pf_v3(0,0,1),.6f);
    auto rotated=E::inverse_inertia(t,rotation,pf_quat_rotate(rotation,input),1e-18f);
    auto expected=pf_quat_rotate(rotation,inverse);
    near(rotated.x,expected.x); near(rotated.y,expected.y); near(rotated.z,expected.z);
    PfVec3 axes[3]={pf_v3(1,0,0),pf_v3(0,1,0),pf_v3(0,0,1)}, half=pf_v3(1,1,1);
    assert(C::overlap(pf_v3(2,0,0),axes,half,axes,half,0));
    assert(!C::overlap(pf_v3(2.01f,0,0),axes,half,axes,half,0));
    assert(C::overlap(pf_v3(2.01f,0,0),axes,half,axes,half,.02f));
    assert(C::face(pf_v3(0,2,0),axes,half,pf_v3(0,0,0),axes,half,pf_v3(0,1,0),0));
    assert(!C::face(pf_v3(0,-2,0),axes,half,pf_v3(0,0,0),axes,half,pf_v3(0,1,0),0));
    assert(!C::face(pf_v3(0,2.01f,0),axes,half,pf_v3(0,0,0),axes,half,pf_v3(0,1,0),0));
    assert(C::face(pf_v3(0,2.01f,0),axes,half,pf_v3(0,0,0),axes,half,pf_v3(0,1,0),.02f));
    PfVec3 diagonal[3]; pf_quat_axes(pf_quat_from_axis_angle(pf_v3(0,0,1),.785398163f),diagonal);
    near(C::radius(diagonal,half,pf_v3(1,0,0)),sqrtf(2));
    assert(C::overlap(pf_v3(2.3f,0,0),axes,half,diagonal,half,0));
    assert(!C::overlap(pf_v3(2.5f,0,0),axes,half,diagonal,half,0));
}
#ifdef __CUDACC__
__global__ void kernel() { check(); }
#endif
int main() {
#ifdef __CUDACC__
    kernel<<<1,1>>>(); auto err=cudaDeviceSynchronize();
    if(err!=cudaSuccess) { fprintf(stderr,"%s\n",cudaGetErrorString(err)); return 1; }
#else
    check();
#endif
    puts("explicit dynamics and clearance analytical checks passed");
}
