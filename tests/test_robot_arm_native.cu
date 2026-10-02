#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <unistd.h>
#include "../ocean/robot_arm/robot_arm.h"
#include "../ocean/robot_arm/robot_arm_model.cuh"
#include "../src/puffysics/native_batch.cuh"
#include "../src/puffysics/model_upload.cuh"
#include "../src/puffysics/model_io.cuh"
#include "../src/puffysics/state_io.cuh"

// CUDA validation of the articulation-only migration. The optional floor
// fixture below is NOT the production robot's collision model.
#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { \
    fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(2); } } while(0)
static void require(bool ok,const char* label) {
    if (!ok) { fprintf(stderr,"FAIL %s\n",label); exit(1); }
}
struct Result { float pose, mass, gravity, jacobian, width_mass, width_step; int failed; };
__device__ static float distance(PfVec3 a,RaVec3 b) {
    return pf_length(pf_sub(a,pf_v3(b.x,b.y,b.z)));
}
__global__ static void compare(PfModel m,PfNativeBatch batch,Result* results) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=batch.count) return;
    PfState& s=batch.states[index]; PfWorkspace& w=batch.workspaces[index]; Result r={};
    RaState old={};
    for(int i=0;i<7;++i) old.q[i]=s.qpos[i]+=0.2f*sinf(index*1.71f+i*2.31f);
    s.qpos[7]=0.004f+0.034f*(0.5f+0.5f*sinf(index*1.33f));
    old.gripper_width=2*s.qpos[7];
    r.failed+=!pf_kinematics(m,s,w);
    RaPose bodies[10],legacy_links[RA_LINKS]; RaVec3 origins[7],axes[7];
    ra_fk(old.q,old.gripper_width,legacy_links,origins,axes,NULL);
    ra_dposes(legacy_links,bodies);
    for(int i=0;i<10;++i) {
        r.pose=fmaxf(r.pose,distance(w.links[i].pose.position,bodies[i].position));
        for(int k=0;k<3;++k) {
            PfVec3 axis=pf_basis(k);
            r.pose=fmaxf(r.pose,distance(pf_quat_rotate(w.links[i].pose.rotation,axis),
                ra_rotate(bodies[i].rotation,ra_v3(axis.x,axis.y,axis.z))));
        }
    }
    float matrix[7][7],gravity[7],armature[7];
    for(int j=0;j<7;++j) armature[j]=0.1f;
    RaDynamics::mass<RaDynamicsModel>(bodies,origins,axes,10,armature,matrix);
    RaDynamics::gravity<RaDynamicsModel>(bodies,origins,axes,10,ra_v3(0,-9.81f,0),gravity);
    pf_articulated_assemble(m,s,w,pf_v3(0,-9.81f,0),1.0f/480);
    for(int i=0;i<7;++i) {
        r.gravity=fmaxf(r.gravity,fabsf(w.rhs[i]-gravity[i]));
        for(int j=0;j<7;++j) r.mass=fmaxf(r.mass,fabsf(pf_mass_entry(m,w,i,j)-matrix[i][j]));
    }
    r.width_mass=fabsf(pf_mass_entry(m,w,7,7)-4*0.0575f);
    r.failed+=!pf_articulated_factor(m,w);
    // Central finite differences, including the follower's contribution.
    for(int d=0;d<8;++d) {
        float q=s.qpos[d],eps=0.0005f;
        PfVec3 analytic[10],minus[10];
        for(int l=0;l<10;++l) analytic[l]=pf_jacobian_entry(m,w,l,d);
        s.qpos[d]=q-eps; pf_kinematics(m,s,w);
        for(int l=0;l<10;++l) minus[l]=w.links[l].pose.position;
        s.qpos[d]=q+eps; pf_kinematics(m,s,w);
        for(int l=0;l<10;++l) r.jacobian=fmaxf(r.jacobian,pf_length(pf_sub(analytic[l],
            pf_scale(pf_sub(w.links[l].pose.position,minus[l]),0.5f/eps))));
        s.qpos[d]=q; pf_kinematics(m,s,w);
    }
    float q=s.qpos[7]; s.qvel[7]=0.123f;
    pf_integrate_coordinates(m,s,0.001f);
    r.width_step=fabsf(s.qpos[7]-(q+0.000123f));
    // Exercise a negative-ratio, offset follower independently of robot poses.
    PfLink links[12]; for(int i=0;i<m.link_count;++i) links[i]=m.links[i];
    links[RA_NATIVE_RIGHT].ratio=-0.75f; links[RA_NATIVE_RIGHT].offset=0.02f;
    PfModel coupled=m; coupled.links=links;
    s.qpos[7]=0.013f; s.qvel[7]=0.2f; pf_kinematics(coupled,s,w);
    PfPose hand=w.links[RA_NATIVE_HAND].pose;
    PfVec3 expected=pf_pose_point(hand,pf_v3(0,-(-0.75f*0.013f+0.02f),0.0584f));
    r.failed+=pf_length(pf_sub(expected,w.links[RA_NATIVE_RIGHT].pose.position))>1e-6f;
    expected=pf_quat_rotate(hand.rotation,pf_v3(0,0.15f,0));
    r.failed+=pf_length(pf_sub(expected,w.links[RA_NATIVE_RIGHT].velocity))>1e-6f;
    PfActuator motor={}; motor.link=RA_NATIVE_RIGHT; motor.kind=PF_MOTOR; motor.gear=2;
    motor.control_min=-10; motor.control_max=10; motor.force_min=-10; motor.force_max=10;
    coupled.actuators=&motor; coupled.actuator_count=1; s.control[0]=3;
    for(int d=0;d<m.nv;++d) w.rhs[d]=w.diagonal[d]=0;
    r.failed+=!pf_actuators(coupled,s,w,0.001f) || fabsf(w.rhs[7]+4.5f)>1e-6f;
    // Save/restore continuation state including external loads and controls.
    float saved[192],restored[192]; double saved_time,restored_time;
    s.time=1.25; s.applied[3]=0.42f; s.force[0]=pf_v3(1,2,3);
    pf_save_state(m,s,saved,&saved_time);
    pf_reset_state(m,s);
    r.failed+=!pf_restore_state(m,s,w,saved,saved_time);
    pf_save_state(m,s,restored,&restored_time);
    r.failed+=restored_time!=saved_time;
    for(size_t k=0;k<pf_state_float_count(m);++k) r.failed+=saved[k]!=restored[k];
    results[index]=r;
}
__global__ static void controls(PfModel m,PfNativeBatch batch) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=batch.count) return;
    PfState& s=batch.states[i];
    for(int a=0;a<m.actuator_count;++a) {
        const PfActuator& actuator=m.actuators[a];
        s.control[a]=actuator.gear*pf_joint_position(m.links[actuator.link],s.qpos);
    }
}
__global__ static void move_targets(PfNativeBatch batch) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=batch.count) return;
    batch.states[i].control[0]+=0.2f;
    batch.states[i].control[1]-=0.1f;
    batch.states[i].control[5]+=0.1f;
    batch.states[i].control[7]=0.02f;
}
struct Summary { int failed; float max_speed, max_arm_error, max_width_error, min_y, max_y; double time; };
__global__ static void summarize(PfModel m,PfNativeBatch batch,Summary* out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=batch.count) return;
    PfState& s=batch.states[i]; PfWorkspace& w=batch.workspaces[i]; Summary r={};
    r.failed=w.status!=PF_OK || !pf_state_valid(m,s); r.time=s.time;
    for(int d=0;d<m.nv;++d) r.max_speed=fmaxf(r.max_speed,fabsf(s.qvel[d]));
    for(int d=0;d<7;++d) r.max_arm_error=fmaxf(r.max_arm_error,fabsf(s.qpos[d]-m.initial_qpos[d]));
    r.max_width_error=fabsf(2*s.qpos[7]-0.08f);
    r.min_y=r.max_y=s.qpos[m.links[RA_NATIVE_OBJECT].qpos+1];
    if(m.link_count>RA_NATIVE_BASE) {
        float y=s.qpos[m.links[RA_NATIVE_BASE].qpos+1];
        r.min_y=fminf(r.min_y,y); r.max_y=fmaxf(r.max_y,y);
    }
    out[i]=r;
}
static void floor_fixture(PfModelStorage& h,RaNativeMode mode) {
    PfGeom ground={}; ground.link=-1; ground.kind=PF_GEOM_PLANE;
    ground.local=pf_pose_identity(); ground.material=pf_default_material();
    ground.type=ground.affinity=1; h.geoms.push_back(ground);
    for(int l=RA_NATIVE_OBJECT;l<(int)h.links.size();++l) {
        PfGeom object=ground; object.link=l;
        object.kind=mode==RA_NATIVE_BASKETBALL?PF_GEOM_SPHERE:PF_GEOM_BOX;
        object.size=mode==RA_NATIVE_BASKETBALL?pf_v3(0.028f,0,0):pf_v3(0.035f,0.035f,0.035f);
        h.geoms.push_back(object); h.pairs.push_back({0,(int)h.geoms.size()-1});
    }
    h.explicit_pairs=true;
}
static size_t bytes_per_world(const PfModel& m,int rows,int contacts) {
    size_t n=m.nv,l=m.link_count;
    return sizeof(PfState)+sizeof(PfWorkspace)+(m.nq+2*n+2*m.actuator_count)*sizeof(float)
        +2*l*sizeof(PfVec3)+(2*(size_t)(m.matrix_size?m.matrix_size:n*n)+3*n+2*rows*n)*sizeof(float)+2*(size_t)m.dependency_count*sizeof(PfVec3)
        +l*sizeof(PfLinkState)+rows*sizeof(PfRow)+contacts*sizeof(PfContact);
}
int main(int argc,char** argv) {
    setvbuf(stdout,nullptr,_IOLBF,0);
    bool bench=argc>1 && !strcmp(argv[1],"--bench");
    bool quick=argc>1 && !strcmp(argv[1],"--quick");
    int count=bench?(argc>2?atoi(argv[2]):4096):quick?8:128;
    require(count>0,"positive world count");
    int threads=bench&&argc>4?atoi(argv[4]):64;
    require(threads==32 || threads==64 || threads==128 || threads==256,"supported block size");
    cudaDeviceProp properties; CUDA(cudaGetDeviceProperties(&properties,0));
    printf("GPU=%s worlds=%d fixture=free_objects_on_plane (no robot collisions)\n",properties.name,count);
    for(int mode=0;mode<3;++mode) {
        PfModelStorage storage; PfModel host={},device={};
        const char* error=ra_native_articulation((RaNativeMode)mode,storage,host);
        if(error) { fprintf(stderr,"%s\n",error); return 1; }
        floor_fixture(storage,(RaNativeMode)mode);
        require(!pf_compile_model(storage,host),"floor fixture compile");
        if(!bench) {
            PfModelStorage bad=storage; PfModel rejected={};
            bad.links[RA_NATIVE_RIGHT].source=RA_NATIVE_RIGHT;
            require(pf_compile_model(bad,rejected)!=nullptr,"reject cyclic mimic");
            bad=storage; bad.links[RA_NATIVE_RIGHT].ratio=0;
            require(pf_compile_model(bad,rejected)!=nullptr,"reject zero mimic scale");
            bad=storage; bad.pairs.push_back(bad.pairs.front());
            require(pf_compile_model(bad,rejected)!=nullptr,"reject duplicate pair");
            bad=storage; bad.pairs={{0,999}};
            require(pf_compile_model(bad,rejected)!=nullptr,"reject invalid pair");
            char path[]="/tmp/robot_native_cache_XXXXXX";
            int fd=mkstemp(path); require(fd>=0,"temporary cache file"); close(fd);
            require(pf_model_write(path,storage,pf_default_step_options()),"write model cache");
            PfModelStorage loaded; PfModel view={}; PfStepOptions saved={};
            require(!pf_model_read(path,loaded,view,saved),"read model cache"); unlink(path);
            require(loaded.explicit_pairs && view.pair_count==host.pair_count
                && view.nv==host.nv && loaded.links[RA_NATIVE_RIGHT].mimic
                && loaded.links[RA_NATIVE_RIGHT].dof==7,"cache preserves coupling and pair policy");
        }
        require(host.nv==(mode==RA_NATIVE_STACK?20:14),"coordinate count");
        require(host.link_count<=12 && pf_state_float_count(host)<=192,"diagnostic buffer capacity");
        require(host.links[RA_NATIVE_LEFT].dof==host.links[RA_NATIVE_RIGHT].dof,"shared width coordinate");
        require(pf_model_upload(host,&device),"upload");
        PfNativeBatch batch={}; require(pf_native_batch_create(device,count,96,32,&batch),"batch allocation");
        cudaStream_t stream; CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        if(!bench) {
            Result* gpu; CUDA(cudaMalloc((void**)&gpu,count*sizeof(Result)));
            compare<<<(count+63)/64,64,0,stream>>>(device,batch,gpu); CUDA(cudaGetLastError());
            std::vector<Result> result(count); CUDA(cudaStreamSynchronize(stream));
            CUDA(cudaMemcpy(result.data(),gpu,count*sizeof(Result),cudaMemcpyDeviceToHost)); CUDA(cudaFree(gpu));
            Result max={};
            for(Result r:result) {
                max.failed+=r.failed; max.pose=fmaxf(max.pose,r.pose); max.mass=fmaxf(max.mass,r.mass);
                max.gravity=fmaxf(max.gravity,r.gravity); max.jacobian=fmaxf(max.jacobian,r.jacobian);
                max.width_mass=fmaxf(max.width_mass,r.width_mass); max.width_step=fmaxf(max.width_step,r.width_step);
            }
            printf("mode=%d comparison failed=%d pose=%g mass=%g gravity=%g jacobian=%g width_mass=%g width_step=%g\n",
                mode,max.failed,max.pose,max.mass,max.gravity,max.jacobian,max.width_mass,max.width_step);
            require(!max.failed && max.pose<3e-6f && max.mass<1e-5f && max.gravity<1e-4f
                && max.jacobian<0.001f && max.width_mass<1e-6f && max.width_step<1e-8f,"articulation comparisons");
        }
        CUDA(pf_native_reset(device,batch,nullptr,stream));
        controls<<<(count+63)/64,64,0,stream>>>(device,batch); CUDA(cudaGetLastError());
        PfStepOptions options={pf_v3(0,-9.81f,0),1.0f/480,1e-5f,16};
        // Short launches permit responsive error reporting; default run is 10s simulated.
        int calls=bench?(argc>3?atoi(argv[3]):100):quick?20:600;
        require(calls>0,"positive call count");
        auto step = [&]() {
            if (!bench) { CUDA(pf_native_step(device,batch,options,8,nullptr,stream)); return; }
            pf_native_step_kernel<<<(count+threads-1)/threads,threads,0,stream>>>(device,batch,options,8,nullptr);
            CUDA(cudaGetLastError());
        };
        if (bench) printf("benchmark_threads=%d\n",threads);
        for(int i=0;i<20;++i) step();
        CUDA(cudaStreamSynchronize(stream));
        cudaEvent_t start,stop; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop));
        std::vector<float> timings;
        int repeats=bench?5:1;
        for(int repeat=0;repeat<repeats;++repeat) {
            CUDA(cudaEventRecord(start,stream));
            for(int i=0;i<calls;++i) step();
            CUDA(cudaEventRecord(stop,stream)); CUDA(cudaEventSynchronize(stop));
            float ms; CUDA(cudaEventElapsedTime(&ms,start,stop)); timings.push_back(ms);
        }
        Summary* gpu; CUDA(cudaMalloc((void**)&gpu,count*sizeof(Summary)));
        summarize<<<(count+63)/64,64,0,stream>>>(device,batch,gpu); CUDA(cudaGetLastError());
        CUDA(cudaStreamSynchronize(stream)); std::vector<Summary> result(count);
        CUDA(cudaMemcpy(result.data(),gpu,count*sizeof(Summary),cudaMemcpyDeviceToHost)); CUDA(cudaFree(gpu));
        Summary max={}; max.min_y=1e30f;
        for(Summary r:result) {
            max.failed+=r.failed; max.max_speed=fmaxf(max.max_speed,r.max_speed);
            max.max_arm_error=fmaxf(max.max_arm_error,r.max_arm_error);
            max.max_width_error=fmaxf(max.max_width_error,r.max_width_error);
            max.min_y=fminf(max.min_y,r.min_y); max.max_y=fmaxf(max.max_y,r.max_y); max.time=r.time;
        }
        printf("mode=%d run failed=%d simulated_s=%.3f speed=%g arm_error=%g width_error=%g object_y=[%g,%g]\n",
            mode,max.failed,max.time,max.max_speed,max.max_arm_error,max.max_width_error,max.min_y,max.max_y);
        require(!max.failed && max.max_speed<1 && max.max_arm_error<0.05f && max.max_width_error<0.002f
            && max.min_y>0.015f && max.max_y<0.04f,"stationary fixture stability");
        std::sort(timings.begin(),timings.end()); float ms=timings[timings.size()/2];
        printf("mode=%d performance median_ms=%.3f min_ms=%.3f max_ms=%.3f calls=%d control_steps_per_s=%.0f substeps_per_s=%.0f batch_control_ms=%.3f allocated_MiB=%.2f\n",
            mode,ms,timings.front(),timings.back(),calls,count*calls*1000.0/ms,count*calls*8000.0/ms,
            ms/calls,bytes_per_world(host,96,32)*count/1048576.0);
        if(!bench && !quick) {
            move_targets<<<(count+63)/64,64,0,stream>>>(batch); CUDA(cudaGetLastError());
            for(int i=0;i<90;++i) CUDA(pf_native_step(device,batch,options,8,nullptr,stream));
            CUDA(cudaMalloc((void**)&gpu,count*sizeof(Summary)));
            summarize<<<(count+63)/64,64,0,stream>>>(device,batch,gpu); CUDA(cudaGetLastError());
            CUDA(cudaStreamSynchronize(stream));
            CUDA(cudaMemcpy(result.data(),gpu,count*sizeof(Summary),cudaMemcpyDeviceToHost)); CUDA(cudaFree(gpu));
            float max_speed=0,width_error=0; int failed=0;
            for(Summary r:result) {
                failed+=r.failed || r.max_arm_error<0.19f || r.max_arm_error>0.21f;
                max_speed=fmaxf(max_speed,r.max_speed);
                width_error=fmaxf(width_error,fabsf(r.max_width_error-0.06f));
            }
            printf("mode=%d moving_targets failed=%d final_speed=%g width_target_error=%g\n",mode,failed,max_speed,width_error);
            require(!failed && max_speed<0.01f && width_error<0.0002f,"arm motion and jaw closure");
        }
        CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(stop)); CUDA(cudaStreamDestroy(stream));
        pf_native_batch_destroy(&batch); pf_model_destroy(&device);
    }
    puts("PASS robot native articulation and floor fixture");
}
