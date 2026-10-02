// g++ -std=c++17 -O2 tests/test_puffysics_impulse_solver.cpp -o /tmp/test_puffysics_impulse_solver
// CUDA: nvcc -x cu -std=c++17 -O3 -arch=sm_86 -cudart shared tests/test_puffysics_impulse_solver.cpp -o /tmp/test_puffysics_impulse_solver_cuda
// This test deliberately includes no environment headers.
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
#include "../src/puffysics/sat_manifold.cuh"
#include "../src/puffysics/impulse_solver.cuh"
using S = PfImpulseSolver;
TEST_DEVICE static S::Config config() {
    S::Config c={};
    c.velocity_iterations=16; c.position_iterations=16;
    c.warm_start=1; c.position_beta=0.8f; c.slop=1e-5f;
    c.speculative_margin=0.0005f; c.restitution_threshold=0.2f;
    c.max_normal_impulse=1e4f; c.max_position_correction=0.01f;
    c.max_position_impulse=1e4f; c.cache_max_age=2;
    return c;
}
TEST_DEVICE static PfBody body(int mode) {
    PfBody b={}; b.mode=mode; b.rotation=pf_quat_identity();
    b.inverse_mass=mode==PF_DYNAMIC?1:0;
    b.inverse_inertia_local=pf_v3(1,1,1);
    return b;
}
TEST_DEVICE static S::Candidate candidate(float separation=0) {
    S::Candidate c={}; c.feature=17;
    c.contact={1,1,separation,pf_v3(0,1,0),pf_v3(0,separation,0),pf_v3(0,0,0)};
    return c;
}
TEST_DEVICE static void rigid_contacts() {
    const float speeds[2]={0.5f,2.0f};
    for(float speed : speeds) {
        PfBody b[2]={body(PF_DYNAMIC),body(PF_STATIC)};
        b[0].linear_velocity=pf_v3(speed,-1,0);
        auto p=candidate(); S::Manifold m={}; S::Cache cache={}; auto c=config();
        assert(S::manifold(0,1,&p,1,0.001f,1,0.25f,0,&m)==1);
        S::solve(b,2,&m,1,0.01f,&c,&cache);
        assert(std::fabs(b[0].linear_velocity.x-(speed<1?0:1.75f))<1e-6f);
        assert(std::fabs(b[0].linear_velocity.y)<1e-6f);
        assert(cache.count==1 && cache.entries[0].feature==17);
        assert(cache.entries[0].normal_impulse>0);
        S::Manifold restored={}; S::manifold(0,1,&p,1,0.001f,1,0.25f,0,&restored);
        S::prepare(b,2,&restored,1,&cache,&c);
        assert(restored.points[0].normal_impulse==m.points[0].normal_impulse);
        assert(std::fabs(restored.points[0].tangent_1_impulse-m.points[0].tangent_1_impulse)<1e-6f);
        for(int i=0;i<3;++i) S::solve(b,2,nullptr,0,0.01f,&c,&cache);
        assert(cache.count==0);
        S::clear_cache(&cache); assert(cache.tick==0 && cache.count==0);
    }
    PfBody b[2]={body(PF_DYNAMIC),body(PF_DYNAMIC)};
    b[0].linear_velocity.y=-1; b[1].linear_velocity.y=1;
    auto p=candidate(); S::Manifold m={}; S::Cache cache={}; auto c=config();
    S::manifold(0,1,&p,1,0.001f,0,0,1,&m);
    S::solve(b,2,&m,1,0.01f,&c,&cache);
    assert(std::fabs(b[0].linear_velocity.y-1)<1e-6f);
    assert(std::fabs(b[1].linear_velocity.y+1)<1e-6f);
}
TEST_DEVICE static void geometry() {
    PfSatShape a={PF_BOX,{pf_v3(0,0,0),pf_quat_identity()},pf_v3(1,1,1)};
    PfSatShape b=a; b.pose.position.x=2.00025f;
    assert(PfSatCollision::query(&a,&b,0.0005f).contact.hit);
    assert(!PfSatCollision::query(&a,&b,0).contact.hit);
    b.pose.position.x=1.5f;
    PfSatCollision::Manifold m={};
    assert(PfSatCollision::manifold(&a,&b,0,&m)==4);
    for(int i=0;i<4;++i) {
        assert(std::fabs(m.point[i].separation+0.5f)<1e-6f);
        assert(m.point[i].normal.x==-1);
        for(int j=0;j<i;++j) assert(m.point_feature[i]!=m.point_feature[j]);
    }
    PfSatShape sphere={PF_SPHERE,{pf_v3(2.25f,0,0),pf_quat_identity()},pf_v3(0.5f,0,0)};
    b=sphere;
    assert(!PfSatCollision::query(&a,&b,0).contact.hit);
    b.pose.position.x=1.25f;
    auto ab=PfSatCollision::query(&a,&b,0).contact;
    auto ba=PfSatCollision::query(&b,&a,0).contact;
    assert(ab.hit && ba.hit && ab.normal.x==-ba.normal.x);
    assert(ab.point_a.x==ba.point_b.x && ab.point_b.x==ba.point_a.x);
    a=b; a.pose.position.x=1.9f;
    assert(PfSatCollision::query(&a,&b,0).contact.hit);
}
// A two-coordinate proxy verifies that reactions need neither seven arm joints
// nor a separate gripper coordinate, and split correction is policy-owned.
struct TwoCoordinateReaction {
    static constexpr bool require_angular_reaction=false;
    struct State { float q[2],v[2]; };
    struct Reaction { int active; float inverse_mass[3]; float j[3][2],response[3][2]; };
    struct AngularReaction { int active; float inverse_mass; };
    struct Displacement { float q[2]; };
    TEST_DEVICE static void apply(const Reaction& r,State& s,int d,float impulse) {
        for(int k=0;k<2;++k) s.v[k]-=r.response[d][k]*impulse;
    }
    TEST_DEVICE static float velocity(const Reaction& r,const State& s,int d) {
        return r.j[d][0]*s.v[0]+r.j[d][1]*s.v[1];
    }
    TEST_DEVICE static void apply_angular(const AngularReaction&,State&,float) {}
    TEST_DEVICE static float angular_velocity(const AngularReaction&,const State&) { return 0; }
    TEST_DEVICE static Displacement correct(State& s,const Reaction& r,float magnitude) {
        Displacement delta={};
        for(int k=0;k<2;++k) { delta.q[k]=-r.response[0][k]*magnitude; s.q[k]+=delta.q[k]; }
        return delta;
    }
    TEST_DEVICE static float displacement(const Reaction& r,const Displacement& delta) {
        return r.j[0][0]*delta.q[0]+r.j[0][1]*delta.q[1];
    }
};
TEST_DEVICE static void articulation() {
    using A=PfImpulseSolverT<PfContactTraits,TwoCoordinateReaction,4,8,16>;
    PfBody b[2]={body(PF_DYNAMIC),body(PF_KINEMATIC)};
    b[0].linear_velocity.y=-1;
    A::Candidate p={}; p.feature=1; p.contact=candidate().contact;
    A::Manifold m={}; A::Cache cache={}; A::Config c={};
    c.velocity_iterations=16; c.max_normal_impulse=1e4f; c.cache_max_age=2;
    A::manifold(0,1,&p,1,0,0,0,0,&m);
    auto& r=m.points[0].reaction;
    r.active=1; r.inverse_mass[0]=0.5f; r.j[0][1]=1; r.response[0][1]=0.5f;
    TwoCoordinateReaction::State state={};
    A::solve(b,2,&m,1,0.01f,&c,&cache,&state);
    assert(std::fabs(b[0].linear_velocity.y+1.0f/3)<1e-6f);
    assert(std::fabs(state.v[1]+1.0f/3)<1e-6f);
    assert(state.v[0]==0);
    // Independently correct penetration without injecting velocity.
    p.contact=candidate(-0.02f).contact;
    A::manifold(0,1,&p,1,0,0,0,0,&m);
    m.points[0].reaction.active=1; m.points[0].reaction.inverse_mass[0]=0.5f;
    m.points[0].reaction.j[0][1]=1; m.points[0].reaction.response[0][1]=0.5f;
    state={}; c.split_position=1; c.position_iterations=16; c.position_beta=0.8f;
    c.max_position_correction=0.01f; c.max_position_impulse=1e4f; c.slop=1e-5f;
    A::prepare(b,2,&m,1,&cache,&c);
    A::solve_positions(b,2,&m,1,&c,&state);
    assert(state.q[1]<0 && state.v[1]==0);
}
TEST_DEVICE static void compliant_and_torsional_contacts() {
    PfBody b[2]={body(PF_DYNAMIC),body(PF_STATIC)};
    auto p=candidate(-0.01f); S::Manifold m={}; S::Cache cache={}; auto c=config();
    S::manifold(0,1,&p,1,0,0,0,0,&m);
    m.points[0].normal_erp=10; m.points[0].normal_cfm=1;
    S::solve(b,2,&m,1,0.01f,&c,&cache);
    assert(std::fabs(b[0].linear_velocity.y-0.05f)<1e-6f);
    // A finite contact patch resists spin even without articulation reactions.
    b[0]=body(PF_DYNAMIC); b[0].linear_velocity.y=-1; b[0].angular_velocity.y=1;
    p=candidate(); p.patch_group=1; p.patch.area=1; p.patch.second_11=p.patch.second_22=0.5f;
    S::manifold(0,1,&p,1,0,1,1,0,&m); S::clear_cache(&cache);
    m.patch_area=1; m.torsional_radius=1;
    S::solve(b,2,&m,1,0.01f,&c,&cache);
    assert(std::fabs(b[0].angular_velocity.y)<1e-6f);
    assert(m.torsional_impulse!=0 && cache.count==2);
}
TEST_DEVICE static void run_tests() {
    geometry(); rigid_contacts(); articulation(); compliant_and_torsional_contacts();
}
#ifdef __CUDACC__
__global__ void test_kernel() { run_tests(); }
#endif
int main() {
#ifdef __CUDACC__
    test_kernel<<<1,1>>>();
    auto status=cudaDeviceSynchronize();
    if(status!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(status)); return 1; }
#else
    run_tests();
#endif
    std::puts("Puffysics generic SAT/impulse solver: PASS");
}
