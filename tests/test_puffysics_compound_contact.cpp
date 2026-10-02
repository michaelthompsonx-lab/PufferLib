// CPU: g++ -std=c++17 -O2 tests/test_puffysics_compound_contact.cpp -o /tmp/test_puffysics_compound_contact
// GPU: nvcc -x cu -std=c++17 -O3 -arch=sm_86 -cudart shared tests/test_puffysics_compound_contact.cpp -o /tmp/test_puffysics_compound_contact_cuda
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
#define TEST_DEVICE __device__
#include "../src/puffysics/compound_contact.cuh"
#include "../src/puffysics/swept_collision.cuh"
using C=PfCompoundContact;
using W=PfSweptCollision;
TEST_DEVICE static PfSatShape box(PfVec3 position,PfVec3 half) {
    return {PF_BOX,{position,pf_quat_identity()},half};
}
TEST_DEVICE static PfBody body(const PfSatShape& s) {
    PfBody b={}; b.mode=PF_KINEMATIC; b.position=s.pose.position; b.rotation=s.pose.rotation;
    return b;
}
TEST_DEVICE static void patch_moments() {
    PfVec3 polygon[4]={pf_v3(-1,2,-2),pf_v3(1,2,-2),pf_v3(1,2,2),pf_v3(-1,2,2)};
    C::Patch p={};
    C::measure_patch(polygon,4,pf_v3(0,0,0),pf_v3(0,1,0),pf_v3(1,0,0),pf_v3(0,0,1),&p);
    assert(std::fabs(p.area-8)<1e-6f);
    assert(p.centroid.x==0 && p.centroid.y==0 && p.centroid.z==0);
    assert(std::fabs(p.second_11-8.0f/3)<1e-5f);
    assert(std::fabs(p.second_22-32.0f/3)<1e-5f && p.second_12==0);
    PfVec3 reversed[4]; for(int i=0;i<4;++i) reversed[i]=polygon[3-i];
    C::Patch q={}; C::measure_patch(reversed,4,pf_v3(0,0,0),pf_v3(0,1,0),pf_v3(1,0,0),pf_v3(0,0,1),&q);
    assert(q.area==p.area && q.second_11==p.second_11 && q.second_22==p.second_22);
    C::measure_patch(polygon,2,pf_v3(0,0,0),pf_v3(0,1,0),pf_v3(1,0,0),pf_v3(0,0,1),&q);
    assert(q.area==0);
}
TEST_DEVICE static void compound_union() {
    auto object=box(pf_v3(0,0.5f,0),pf_v3(0.5f,0.5f,0.5f));
    PfBody moving=body(object);
    PfSatShape components[6]; PfBody motions[6];
    for(int i=0;i<6;++i) { components[i]=box(pf_v3(0,-0.1f,0),pf_v3(0.5f,0.1f,0.5f)); motions[i]=body(components[i]); }
    C::Candidate scratch[C::candidate_capacity]; C::Manifold m={}; C::Options o;
    auto result=C::box(object,moving,components,motions,6,pf_v3(0,1,0),o,scratch,0,1,&m);
    assert(result.status==C::contact && result.point_count==4);
    // Six overlapping components form one exposed square, with no double-counted area.
    assert(result.component_mask==1 && std::fabs(m.patch_area-1)<1e-6f);
    assert(std::fabs(m.patch_second_moment-1.0f/6)<1e-6f);
    for(int i=0;i<m.point_count;++i) {
        assert(m.points[i].patch_group==1);
        assert((m.points[i].feature&0xf0000000u)!=0xa0000000u);
        for(int j=0;j<i;++j) assert(m.points[j].feature!=m.points[i].feature);
    }
    // Adjacent rectangles share a seam; the exposed union retains its full area.
    components[0]=box(pf_v3(-0.25f,-0.1f,0),pf_v3(0.25f,0.1f,0.5f));
    components[1]=box(pf_v3(0.25f,-0.1f,0),pf_v3(0.25f,0.1f,0.5f));
    motions[0]=body(components[0]); motions[1]=body(components[1]);
    result=C::box(object,moving,components,motions,2,pf_v3(0,1,0),o,scratch,0,1,&m);
    assert(result.status==C::contact && result.component_mask==3);
    assert(std::fabs(m.patch_area-1)<1e-6f);
    assert(std::fabs(m.patch_second_moment-1.0f/6)<1e-6f);
    // An elevated component hides the lower component's entire support surface.
    components[0]=box(pf_v3(0,-0.1f,0),pf_v3(0.5f,0.1f,0.5f));
    components[1]=components[0]; components[1].pose.position.y=0;
    motions[0]=body(components[0]); motions[1]=body(components[1]);
    result=C::box(object,moving,components,motions,2,pf_v3(0,1,0),o,scratch,0,1,&m);
    assert(result.status==C::contact && result.component_mask==2);
    assert(std::fabs(m.patch_area-1)<1e-6f);
    // Common rotation changes the surface normal without changing its area.
    PfQuat rotation=pf_quat_from_axis_angle(pf_v3(0,0,1),0.7f);
    object.pose.position=pf_quat_rotate(rotation,object.pose.position); object.pose.rotation=rotation;
    moving=body(object);
    components[0].pose.position=pf_quat_rotate(rotation,components[0].pose.position); components[0].pose.rotation=rotation;
    motions[0]=body(components[0]);
    result=C::box(object,moving,components,motions,1,pf_quat_rotate(rotation,pf_v3(0,1,0)),o,scratch,0,1,&m);
    assert(result.status==C::contact && std::fabs(m.patch_area-1)<2e-6f);
    // Count/capacity errors are explicit and do not publish a partial manifold.
    using Small=PfCompoundContactT<PfContactTraits,PfImpulseSolver,2,1>;
    Small::Candidate tiny[1]; Small::Options so; Small::Manifold sentinel={}; sentinel.body_a=123;
    auto small=Small::box(object,moving,components,motions,1,pf_quat_rotate(rotation,pf_v3(0,1,0)),so,tiny,0,1,&sentinel);
    assert(small.status==Small::capacity && sentinel.body_a==123);
    result=C::box(object,moving,components,motions,0,pf_v3(0,1,0),o,scratch,0,1,&m);
    assert(result.status==C::invalid);
    o.group_base=UINT32_MAX;
    result=C::box(object,moving,components,motions,1,pf_v3(0,1,0),o,scratch,0,1,&m);
    assert(result.status==C::invalid);
}
TEST_DEVICE static void speculative_sphere() {
    PfSatShape component=box(pf_v3(0,-0.1f,0),pf_v3(0.5f,0.1f,0.5f));
    PfBody component_body=body(component);
    PfSatShape sphere={PF_SPHERE,{pf_v3(0,0.26f,0),pf_quat_identity()},pf_v3(0.25f,0,0)};
    PfBody moving=body(sphere); C::Options o; o.dt=0.02f;
    C::Manifold m={};
    auto result=C::sphere(sphere,moving,&component,&component_body,1,pf_v3(0,1,0),o,0,1,&m);
    assert(result.status==C::no_contact);
    moving.linear_velocity.y=-1;
    result=C::sphere(sphere,moving,&component,&component_body,1,pf_v3(0,1,0),o,0,1,&m);
    assert(result.status==C::contact && result.component_mask==1 && result.point_count==1);
    assert(m.body_a==0 && m.body_b==1 && m.points[0].separation>0);
    moving.linear_velocity.y=1;
    result=C::sphere(sphere,moving,&component,&component_body,1,pf_v3(0,1,0),o,0,1,&m);
    assert(result.status==C::no_contact);
}
TEST_DEVICE static void swept_queries() {
    PfVec3 zero=pf_v3(0,0,0);
    auto a=box(pf_v3(-2,0,0),pf_v3(0.5f,0.5f,0.5f));
    auto b=box(zero,pf_v3(0.5f,0.5f,0.5f));
    auto s=W::shape(&a,&b,pf_v3(2,0,0),zero,zero,zero,1,0);
    assert(s.hit && std::fabs(s.toi-0.5f)<1e-6f);
    assert(W::approaching_time(s,pf_v3(2,0,0),1)==s.toi);
    assert(W::approaching_time(s,pf_v3(-2,0,0),1)==1);
    assert(!W::shape(&a,&b,pf_v3(2,0,0),zero,zero,zero,0.25f,0).hit);
    assert(!W::shape(&a,&b,pf_v3(-2,0,0),zero,zero,zero,1,0).hit);
    a.type=PF_SPHERE;
    s=W::shape(&a,&b,pf_v3(2,0,0),zero,zero,zero,1,0);
    assert(s.hit && std::fabs(s.toi-0.5f)<1e-6f);
    b.type=PF_SPHERE;
    s=W::shape(&a,&b,pf_v3(2,0,0),zero,zero,zero,1,0);
    assert(s.hit && std::fabs(s.toi-0.5f)<1e-6f);
    a.pose.position.x=-0.9f;
    s=W::shape(&a,&b,pf_v3(-2,0,0),zero,zero,zero,1,0);
    assert(s.hit && s.toi==0 && W::approaching_time(s,pf_v3(-2,0,0),1)==1);
    W::Ring ring={zero,pf_v3(1,0,0),pf_v3(0,0,1),pf_v3(0,1,0),1,0.1f};
    s=W::sweep_sphere_ring(pf_v3(1,0.5f,0),pf_v3(0,-1,0),0.1f,ring,1);
    assert(s.hit && std::fabs(s.toi-0.3f)<1e-6f);
    assert(!W::sweep_sphere_ring(pf_v3(1,0.5f,0),zero,0.1f,ring,1).hit);
    ring.axis_z=pf_v3(0,1,0); ring.normal=pf_v3(0,0,1);
    s=W::sweep_sphere_ring(pf_v3(1,0,0.5f),pf_v3(0,0,-1),0.1f,ring,1);
    assert(s.hit && std::fabs(s.toi-0.3f)<1e-6f);
    auto contact=W::sphere_ring(zero,0.1f,ring,0);
    assert(std::isfinite(contact.separation) && std::fabs(contact.separation-0.8f)<1e-6f);
    contact=W::sphere_ring(pf_v3(1,0,0),0.1f,ring,0);
    assert(contact.hit && contact.normal.z==1);
}
TEST_DEVICE static void run_tests() { patch_moments(); compound_union(); speculative_sphere(); swept_queries(); }
#ifdef __CUDACC__
__global__ void test_kernel() { run_tests(); }
#endif
int main() {
#ifdef __CUDACC__
    test_kernel<<<1,1>>>(); auto status=cudaDeviceSynchronize();
    if(status!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(status)); return 1; }
#else
    run_tests();
#endif
    std::puts("Puffysics compound contacts and swept queries: PASS");
}
