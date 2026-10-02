# Robot arm physics extraction

## Environment line-count ledger

Counts include all production source in `ocean/robot_arm`, including adapters;
tests, documentation and Puffysics engine files are excluded. Blank lines and
comments are counted, consistently with `wc -l`.

| Checkpoint | CUDA header | Physics adapters | All environment source | Stage reduction | Cumulative reduction |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before migration | 3,113 | 0 | 5,553 | — | — |
| Stage 1: SAT and impulse solver | 1,576 | 105 | 4,121 | 1,432 | 1,432 |
| Stage 2: compound contacts and swept queries | 899 | 124 | 3,463 | 658 | 2,090 (37.6%) |
| Stage 3: serial articulation dynamics | 899 | 150 | 3,379 | 84 | 2,174 (39.2%) |
| Stage 4: shared preparation and remaining helpers | 853 | 156 | 3,219 | 160 | 2,334 (42.0%) |
| Stage 5: SKILL_ISSUES environment cleanup | 1,038 | 0 | 3,124 | 95 | 2,429 (43.7%) |
| Geometry shadows, lighting and fork build compatibility | 1,038 | 0 | 3,200 | -76 | 2,353 (42.4%) |

Reproduce the current count with `wc -l ocean/robot_arm/*`. Stage 5 folds the
adapters into the existing headers; their definitions remain included in the
total. Earlier stage descriptions retain their historical filenames.

## Stage 1: SAT and impulse solver

The first production migration extracts margin-aware box/sphere SAT, clipped
manifolds and the persistent impulse solver from `robot_arm_cuda.cuh` into
Puffysics. The environment calls these engine modules through a 105-line adapter.
Its CUDA header shrinks from 3,113 to 1,576 lines: the environment loses 1,432
lines after accounting for the adapter.

The engine modules have no environment dependencies or assumed joint count:

- `src/puffysics/sat_manifold.cuh`: `PfSatCollision::query()` and `manifold()`;
  box/box, sphere/box and sphere/sphere queries, speculative margin, B-to-A
  normals, clipped four-point manifolds and stable feature IDs.
- `src/puffysics/impulse_solver.cuh`: `PfImpulseSolver::manifold()`, `sort()`
  and `solve()`; feature-keyed warm starting, static/sliding friction,
  restitution, compliant normal rows, split correction and contact-patch torsion.
- `src/puffysics/contact_traits.cuh`: default adapters for `PfBody`, `PfVec3`
  and Puffysics `wxyz` quaternions, plus an optional articulation policy.

These APIs are also included by `puffysics.cuh`. They are explicit device
entry points; existing `pf_step()` and `pf_articulated_step()` do not silently
switch solvers. The native articulated row solver remains a separate path.

## Calling the solver from another environment

Use `PfSatShape` with box half-extents or a sphere radius in `half_extents.x`.
Transform it to world space before querying. For each SAT manifold, copy its
contacts and `point_feature` IDs into `PfImpulseSolver::Candidate` entries,
then call `manifold()` with the body indices and material coefficients. Contacts
from other geometry generators can supply candidates directly.

Zero-initialize storage. Retain `PfImpulseSolver::Cache` across substeps and
call `clear_cache()` on episode reset or body/pair topology changes. Bodies and
contact buffers remain caller-owned. Call `sort()` before `solve()` for stable
pair ordering. `solve()` performs contact preparation, optional split correction,
velocity solving and cache writes; external forces, collision generation and
position integration remain the caller's responsibility.

Supply a `Config` explicitly. Iteration counts are capped at 64. Per-manifold
friction/restitution coefficients are authoritative; the legacy coefficient
fields in Config are retained for layout compatibility. Use a finite positive
dt, nonnegative limits and cache age, valid body indices, normalized rotations,
and finite nonnegative masses/inertias. Static-body velocities must be zero;
kinematic velocities are prescribed. Normal rows use `normal_erp` in inverse
seconds and `normal_cfm` as inverse effective mass; zero means a hard row.

Default capacities are 48 manifolds, 20 input candidates per manifold, four
selected contact points and 192 cache entries. Another environment can choose
its capacities with `PfImpulseSolverT<PfContactTraits, PfNoContactReaction,
MaxManifolds, MaxCandidates, MaxCache>`. Input candidates and manifold counts
are bounded by these capacities; callers must size storage accordingly and
handle their own contact overflow. More candidates than capacity are truncated.

A patch supplies area and planar central second moments. Points sharing one
physical patch use the same nonzero `patch_group`. Supply the manifold's patch
area and positive torsional radius to enable torsion. Translational friction
consumes part of the patch's torsional friction budget. The rigid-body default
supports torsion without an articulation reaction.

For articulated proxies, instantiate the solver with another reaction policy.
The policy owns the coordinate count, Jacobians, responses, live velocity access
and split-displacement updates. See `PfNoContactReaction` for the interface and
`TwoCoordinateReaction` in the standalone tests for a working two-coordinate
example. `require_angular_reaction` controls whether torsion requires a proxy
reaction or can also act between ordinary rigid bodies. Active reactions require a non-null policy state. Reactions represent
body B; body A uses normal rigid-body dynamics. Avoid also giving body B a
rigid dynamic response if its reaction already represents that mass.

## Robot adapter after stage 1

`robot_arm_physics.cuh` preserves the environment's `xyzw` layout, exact math
operations and seven-arm-coordinate/full-jaw-width reactions. Split correction
retains the legacy arm **velocity** update and jaw-width clamp. Correcting that
existing behavior is a separate dynamics change, rather than part of extraction.

The environment still owns robot model data, smooth articulated dynamics,
compound-pad exposed-surface generation, patch measurements, continuous collision
queries, table penetration guards and basketball rim/rebound handling. Task
state, actions, observations, rewards, resets and rendering remain there too.

Compound contact geometry and swept queries are now extracted in stage 2 below.
The next stage is native articulation integration, which needs explicit validation
of different native servo integration, velocity-dependent forces, arm/jaw
coupling and joint-limit handling.

## Verification

Standalone analytical tests include no robot headers. The same suite runs on
CPU and on a CUDA device: margin queries, box patches, reversed sphere/box
contacts, rigid friction/restitution, cache restoration/expiry/reset, compliant
rows, independent-body torsion and a two-coordinate articulated proxy.

```sh
g++ -std=c++17 -O2 tests/test_puffysics_impulse_solver.cpp \
  -o /tmp/test_puffysics_impulse_solver
/tmp/test_puffysics_impulse_solver
nvcc -x cu -std=c++17 -O3 -arch=sm_86 -cudart shared \
  tests/test_puffysics_impulse_solver.cpp -o /tmp/test_puffysics_impulse_solver_cuda
env LD_LIBRARY_PATH=/run/opengl-driver/lib /tmp/test_puffysics_impulse_solver_cuda
```

`tests/test_robot_arm_physics.cu` captures all RaState fields, observations,
rewards, terminals and contact counts for 32 worlds per mode over 240 control
steps. Half the worlds start with objects at the fingers; scripted actions move
the arm and close/open the jaw. Compile the same harness with `ROBOT_HEADER`
pointing at a saved old CUDA header to compare pre-extraction output. The saved
header also needs its companion task/render/core headers. Outputs are raw binary
captures for matching builds, not portable serialization.

```sh
nvcc -std=c++17 -O3 -lineinfo -arch=sm_86 -cudart shared \
  -I src -I raylib-5.5_linux_amd64/include \
  tests/test_robot_arm_physics.cu -o /tmp/ra-migration-rollout
env LD_LIBRARY_PATH=/run/opengl-driver/lib \
  /tmp/ra-migration-rollout /tmp/ra-migration-states.bin
```

Validation on 2026-10-02: CPU and GPU standalone analytical suites passed, as
did the existing rigid-contact CPU suite and native robot articulation quick
suite. Production old/new captures matched byte for byte over 23,040 world
transitions, with 32,274 captured contact manifolds and zero nonfinite outputs.
Both 18,339,840-byte captures had SHA-256
`c7c77499408e3d823b6a82c240d4a1adf5a2d6f4b4e9d76e1189762636c4e54a`.
This establishes parity for this scripted workload and build, not task success
or parity for every possible trajectory. No throughput improvement is claimed.

CUDA Compute Sanitizer memcheck of the standalone device suite reported zero errors.

The production CUDA trainer build also passed:
`bash ./build.sh robot_arm /tmp/robot-arm-puffysics-build`.

## Stage 2: compound contacts and swept queries

`compound_contact.cuh` adds `PfCompoundContactT`, defaulting to native Puffysics
shapes/bodies and eight components. It computes the exposed surface of overlapping
rectangular support components, clips box contacts against visible cells,
measures patch area/centroid/central second moments, reduces candidates to four
points and assembles patch torsion data. Sphere contacts select one eligible
component using normal alignment and speculative point velocity. Geometry,
component count, velocities, material values, tolerances and feature namespaces
are supplied by the caller; there is no robot or five-component assumption.

All rectangular components must share a local X/Z frame, with the supplied unit
surface normal along either local Y direction. This models one exposed support
surface of an aligned-box union; it is not arbitrary CSG or a full compound-solid
collision detector. Common world rotation is supported. Box contacts map to one
representative body B, so components must belong to the same rigid/articulated
support. Sphere contacts map to `body_b + component_index`, requiring consecutive
component body indices. The caller supplies those indices and motion arrays.

`PfCompoundContactT<T, Solver, MaxComponents, MaxCandidates, Features>` supports
up to 32 components with a 32-bit component mask. Runtime component counts can
be smaller than capacity. Allocate `Candidate[candidate_capacity]` scratch for
box contacts; sphere contacts do not need that scratch. The returned `Result`
distinguishes contact, no contact, invalid component count/kind/group range,
and candidate-capacity overflow. An overflowing candidate buffer publishes no
partial manifold. Polygon clipping retains the existing eight-vertex bound.
Per-component patches are merged once per selected patch group; the final
summary covers groups represented by the reduced manifold.

`Options` controls timestep, margin, friction, restitution, boundary/support
tolerances, normal alignment, feature namespace and patch-group base. Inputs
must be finite, geometry valid, timestep positive and tolerances nonnegative.
Keep component ordering stable while retaining cached impulses. Default feature
IDs hash the namespace, component, cell and clipping feature, reserving the
angular-cache namespace; patch ownership
is carried separately in `patch_group`, with no assumed feature bit layout.
A caller can provide another feature policy. `robot_arm` supplies its previous
encoding so existing cache IDs and tie-breaking stay unchanged.

`swept_collision.cuh` adds `PfSweptCollisionT::shape()`, `sphere_ring()`,
`sweep_sphere_ring()` and `approaching_time()`. Box/sphere queries use conservative
advancement under constant world linear/angular velocities, including rotational
radius bounds. Rings accept a center, orthonormal plane frame, major radius and
tube radius, allowing arbitrary placement/orientation. The current sphere-ring
sweep supports a stationary ring and linearly moving sphere. Both sweep routines
retain the 12-iteration budget, 1e-8 speed threshold and 1e-7 advance threshold.
A reported miss can mean the iteration budget was exhausted; these are the
existing approximate queries, not an exact continuous collision guarantee.
Use the iteration count/last contact for any stricter caller policy. Only box
and sphere shapes are supported by the shape sweep. These APIs do not apply
impulses or advance the caller's state.

Production now calls these engine functions through `ra_pad_contact()` and
small ring-geometry wrappers. It retains scene pair selection, pad material
compliance setup, reaction Jacobians, task state and body/joint integration.
Stage 2 reduces the CUDA header by 677 lines and grows the adapter by 19 lines,
for a net environment reduction of 658 lines.

The independent `tests/test_puffysics_compound_contact.cpp` suite runs on CPU
and CUDA. It checks polygon moments and winding, six-component overlap without
area duplication, adjacent seams, hidden surfaces, common rotation, candidate
capacity reporting, speculative sphere motion, analytic box/sphere TOIs,
approaching/separating contacts, and rings in two different planes. CUDA
Compute Sanitizer memcheck reports zero errors.

```sh
g++ -std=c++17 -O2 tests/test_puffysics_compound_contact.cpp \
  -o /tmp/test_puffysics_compound_contact
/tmp/test_puffysics_compound_contact
nvcc -x cu -std=c++17 -O3 -arch=sm_86 -cudart shared \
  tests/test_puffysics_compound_contact.cpp -o /tmp/test_puffysics_compound_contact_cuda
env LD_LIBRARY_PATH=/run/opengl-driver/lib /tmp/test_puffysics_compound_contact_cuda
```

The production 23,040-transition captures again match stage 1 byte for byte:
32,274 captured manifolds, zero nonfinite outputs and the same SHA-256 listed
above. `tests/test_robot_arm_collision_capture.cu` additionally captures focused
compound manifolds, patch moments, features, component masks, shape sweeps and
ring contacts/sweeps over 128 varied fixtures. Compile it against a saved stage-1
header with `ROBOT_HEADER` and `ROBOT_ARM_STAGE1` to compare the previous API.

The focused captures match byte for byte across all 128 fixtures: 181 compound
manifolds, 125 shape-sweep hits and 98 ring-sweep hits. Both 416,768-byte files
have SHA-256
`cefe34198511925f90d741b782c84e78ade8a1a7ca69fb7de96d3089af30489c`.
The ring query keeps the original arithmetic in the common world-X/Z frame;
other ring orientations use the supplied plane frame. These comparisons cover
sampled queries and trajectories, rather than proving parity for every state.

The final stage-2 production trainer build passed with
`bash ./build.sh robot_arm /tmp/robot-arm-puffysics-build`, and the final
23,040-transition capture retained the same byte-for-byte match.

## Stage 3: serial articulation dynamics

`src/puffysics/serial_dynamics.cuh` provides `PfSerialDynamicsT<N, Traits>`:
full-tensor joint mass assembly, gravity projection, regularized Cholesky,
forward/back substitution, contact Jacobians and inverse effective mass,
point velocity and angular velocity. The default traits use native Puffysics
vectors and quaternions. Joint count, body count, armature, gravity, poses and
model parameters are supplied by the caller; there are no robot-specific
constants in the engine.

This API handles serial revolute chains with fixed attachments. Each body's
`Model::last(body)` identifies the last influencing joint; all earlier joints
influence that body. Use -1 for a fixed root. Valid indices are [-1, N-1],
body count must be nonnegative, axes must be unit vectors, and buffers must
match the supplied dimensions. Model methods provide mass, local center of
mass and the six entries of the symmetric local inertia tensor. Poses use
`position` and `rotation` fields. Traits supply vector arithmetic and quaternion
rotation/conjugation. Matrix inputs and outputs must use separate buffers.

`factor()` reads the lower triangle and writes only the lower triangle;
`solve()` consumes it. The caller chooses a positive diagonal floor (default
1e-8), preserving the production regularization policy. Factorization does not
report invalid/nonfinite matrices; callers must supply finite physical data,
nonnegative armature and a positive-definite mass matrix. Regularizing a small
pivot does not establish that an invalid model is physically meaningful.

Production keeps its model data and FK. A 26-line dynamics adapter preserves
its xyzw quaternion arithmetic, while the contact setup, body velocities and
throw-release velocity call the engine directly. `robot_arm.h` shrinks from
1,049 to 939 lines. Including the new adapter, this stage removes 84 environment
lines: 3,463 to 3,379, for a cumulative reduction of 2,174 lines (39.2%).

The explicit PD motor updates, separate jaw dynamics, joint limits and table
collision guard remain in the environment. Switching to the compiled native
articulation would also introduce velocity bias, implicit actuator damping and
arm/jaw coupling; this stage preserves the existing physical behavior rather
than introducing those changes during extraction.

Independent `tests/test_puffysics_serial_dynamics.cpp` checks a two-joint chain
against analytical mass/gravity values, linear solve residuals, linear/angular
contact effective mass, fixed-root behavior, point/angular velocities, rotated
full inertia tensors and caller-selected pivot regularization. It runs on CPU
and CUDA; CUDA Compute Sanitizer memcheck reports zero errors.

```sh
g++ -std=c++17 -O2 tests/test_puffysics_serial_dynamics.cpp \
  -o /tmp/test_puffysics_serial_dynamics
/tmp/test_puffysics_serial_dynamics
nvcc -x cu -std=c++17 -O3 -arch=sm_86 -cudart shared \
  tests/test_puffysics_serial_dynamics.cpp -o /tmp/test_puffysics_serial_dynamics_cuda
env LD_LIBRARY_PATH=/run/opengl-driver/lib /tmp/test_puffysics_serial_dynamics_cuda
```

The final production trainer build passed with
`bash ./build.sh robot_arm /tmp/robot-arm-puffysics-build`. The final migrated
23,040-transition GPU capture matches the saved stage-2 capture byte for byte:
32,274 manifolds, zero nonfinite outputs, and SHA-256
`c7c77499408e3d823b6a82c240d4a1adf5a2d6f4b4e9d76e1189762636c4e54a`.
The comparison covers all captured state, observation, reward, terminal and
contact-count data across picking, stacking and basketball. It establishes
sampled trajectory parity, not equivalence for every possible state.

## Stage 4: shared preparation and remaining helpers

Production preparation now computes FK once per substep and uses its local
links, origins and axes for both mass and gravity assembly. It then copies the
same geometry into staged storage for body setup and contact responses.
This removes two FK evaluations per substep (16 per eight-substep control
step). The table guard and post-integration FK remain separate: their joint
positions and jaw widths can differ. Joint-axis quaternion data still uses a
local array in FK, and shared preparation retains local buffers. These choices
preserve CUDA floating-point evaluation; direct per-joint parameter access or
direct assembly from global staged buffers changed rounding in comparisons.

`explicit_dynamics.cuh` introduces `PfExplicitDynamicsT<Traits>` with clamped
explicit PD force, limited velocity updates, joint/slide position integration,
spring/damper compliance, box/sphere inertia and full-tensor inverse inertia.
Parameters are supplied by the caller; the engine has no robot dimensions,
control gains or limit-response constants. Production passes its existing
0.15 joint-limit response, separate full-width jaw mass and material constants.
Inputs must be finite with valid physical inertias, ordered limits,
nonnegative force/speed limits and coefficients. The helpers do not validate
models or apply implicit motors. Joint integration retains the strict limit
crossing test; slides also stop outward motion exactly at a boundary.

`box_clearance.cuh` introduces `PfBoxClearanceT<Traits>::overlap()` and `face()`.
The overlap predicate tests the 15 box SAT axes with per-axis margin and skips
cross axes shorter than the supplied positive epsilon (default 1e-6).
The face predicate adds the inward-side and exposed-face separation checks.
Callers supply orthonormal axes, positive extents, nonnegative margin and an
inward normal equal to either sign of box B's local-Y axis. Production's task
clearance wrapper tests each pad component and returns a boolean. It no longer
computes witness-point patch averages that neither task call site consumed.
Solver manifold generation and task clearance remain distinct predicates.

Contact telemetry and active-pad detection now share one manifold/point pass,
after final FK so wrist moment arms keep their previous reference point.
A shared pad-side helper replaces repeated body-range classifications, and
pad patch-group membership is counted with each point pair once. Cube pad
queries are common to all modes, with additional base queries for stacking.
Unused `pad_face` metadata and two obsolete capacity macros are removed.

`robot_arm_parameters.h` is the single source for arm masses, centers, full
inertias, joint offsets/rest rotations, limits and home coordinates. Production
and the native fixture consume it with explicit quaternion-layout conversion.
Fixed finger-pad and gripper-shell geometry now uses local data tables, keeping
its previous index fallback behavior. Robot-specific data stays in the
environment and is included in the LOC ledger.

The environment drops from 3,379 to 3,219 lines, including the new 74-line
parameter header and the enlarged dynamics adapter. Net reduction for this
stage is 160 lines; the cumulative reduction is 2,334 lines (42.0%).

The independent `tests/test_puffysics_explicit_dynamics.cpp` covers saturated
and unsaturated motors, velocity and coordinate limits, exact-boundary joint
versus slide behavior, compliance, primitive/full-tensor inertia and rotated
box clearance. It runs on CPU and CUDA, with zero Compute Sanitizer memcheck
errors. The existing serial dynamics, impulse and compound-contact CPU tests
also pass.

`tests/test_robot_arm_clearance_capture.cu` compares 4,096 varied fixtures,
including near-parallel orientations and zero-margin face contacts, recording
two clearance decisions and every pad/shell pose and extent. Its 995 hits and
all geometry data match the saved stage-3 implementation byte for byte.
Both captures have SHA-256
`4c8f30906008b5bf61a2fd82a77c523b21a2de06692be218999dd14652be352c`.
Compile the saved implementation with `ROBOT_HEADER` and `ROBOT_ARM_STAGE3`.

The production trainer build and the rebuilt native articulation fixture pass.
The final 23,040-transition capture has 32,268 manifolds and zero nonfinite
outputs. It is not byte-identical to stage 3: the first difference is one ULP
in a joint velocity. In the sampled capture, every categorical state field
(including RNG, task flags and counters) and every terminal value is unchanged.
Maximum absolute reward difference is 0.00181713; maximum episode-return
difference is 0.00648356. Joint positions differ by at most 0.00318593 rad,
and long contact trajectories diverge, with a maximum componentwise object
position difference of 0.178303 m. The intentionally contact-heavy initial
fixtures amplify small floating-point differences. These results support
sampled task-state equivalence and mathematical equivalence of the extracted
updates, not strict trajectory parity or policy/training equivalence.

Changing launch size alone is byte-identical within the final implementation:
32- and 128-thread launches produce the same complete 23,040-transition capture,
SHA-256 `1f0936dfad2c21ba87a949a8e275ca6f47ec32143e7ce89966253191cb69b3b9`.
A second capture uses 65 worlds to exercise partially filled blocks; its 46,800
transitions and 65,521 manifolds also match across both launch sizes, with zero
nonfinite outputs and SHA-256
`b2d0c2f01854f470424cae9a1689b2dc1f0e74c839ea29239e6f2755e37b9fd1`.

### GPU launch optimization

The production default is now 32 threads per block. The physics kernel uses
255 registers per thread and no shared memory. A 4,096-world batch with the
previous 128-thread layout launches only 32 blocks on the RTX 3080 Ti's 80 SMs;
32-thread blocks distribute those worlds across more SMs. Source-level FK
sharing by itself did not establish a runtime win: the first 128-thread timing
was 1–2% slower, and per-thread local storage grew from 3,408 to 3,744 bytes.
The measured gain comes from launch layout. Per-world storage remains 102,944
bytes; cache and scratch capacities are unchanged.

`tools/bench_robot_arm.cu` now accepts `[worlds] [calls] [threads]` and reports
kernel registers/local storage. Each result below is the median of five
100-call timings, each after 20 warmup calls, using production begin/physics/
finish kernels with zero arm actions and an open gripper. No policy inference,
rendering or trainer copies are included. CUDA, C++17, -O3 -lineinfo -arch=sm_86;
GPU: RTX 3080 Ti. All measured outputs are finite. The comparison uses the same
final refactored physics code at each launch size.

| Worlds | Mode | 128 threads, batch control ms | 32 threads, batch control ms | Throughput gain |
| ---: | --- | ---: | ---: | ---: |
| 512 | Picking | 4.658 | 2.248 | 107.2% |
| 512 | Stacking | 8.681 | 4.328 | 100.6% |
| 512 | Basketball | 2.664 | 1.213 | 119.6% |
| 4,096 | Picking | 6.063 | 5.025 | 20.7% |
| 4,096 | Stacking | 11.562 | 10.095 | 14.5% |
| 4,096 | Basketball | 3.331 | 3.001 | 11.0% |
| 8,192 | Picking | 9.374 | 8.648 | 8.4% |
| 8,192 | Stacking | 18.362 | 16.777 | 9.4% |
| 8,192 | Basketball | 5.567 | 5.291 | 5.2% |

At 4,096 worlds, 64 threads also improved on 128, but 32 was faster in all three
modes. 256 threads were slower. These measurements do not establish gains on
other GPUs, larger batches or contact-heavy trained-policy workloads.
GPU storage layout and broader collision pruning remain profiling follow-ups.


## Stage 5: SKILL_ISSUES environment cleanup

Applied the repository's `SKILL_ISSUES.md` guide to the existing environment.
Production source falls from 3,219 to 3,124 lines, including the formatting
pass: 95 lines removed, 2,429 fewer than the original environment (43.7%).
The source remains seven files; no new environment files or engine APIs were
introduced. Puffysics itself is unchanged in this stage.

- Fold the physics and dynamics adapters into `robot_arm_cuda.cuh` and
  `robot_arm.h`, removing their two separate headers. Existing engine traits
  and template interfaces remain necessary to call the shared Puffysics APIs;
  redesigning those interfaces is outside this environment cleanup.
- Remove `ra_dpose`, `ra_massm` and `ra_gravt` from production source. The native
  test prepares its reference FK/body poses and calls the shared mass/gravity
  routines directly, with the same analytical comparisons and tolerances.
- Fold single-use model accessors, interpolation/support wrappers, contact
  telemetry, object integration, grasp bookkeeping and robot contact collection
  into their consumers. Keep helpers used by multiple callers, including the
  focused collision fixtures, and the existing kernel stages.
- Accumulate energy/return and previous actions once in the finish kernel,
  before terminal logging and reset, rather than in both task reward functions.
- Classify each solved manifold's pad side once outside its point loop.
  Trial consolidation of the two body/shape counts changed the world stride
  and showed repeated picking/stacking slowdowns. Retain the original count
  fields and binding layout; the original 102,944-byte world allocation is
  preserved instead of trading layout performance for five source lines.
- Merge object-integration branches with the same position/rotation update and
  start table-contact loops at their first participating body.
- Assert internal geometry index and compound-result invariants instead of
  silently clamping invalid indices or treating API errors as contact misses.
  Keep finite-capacity manifold guards: they protect storage and are separate
  from invalid-input handling. Asset-loading fallbacks remain supported.
- Embed renderer storage in the existing host struct, eliminating its lazy
  heap allocation. Fold its single-use draw/close bodies into the host entry
  points, retaining model/shader unloading and camera/render behavior.
- Apply four-space indentation, four-space continuations, a 100-column maximum,
  braced multi-line conditionals/loops, and compact readable constant tables.

### Validation

The production trainer builds successfully. The updated native articulation
fixture passes all 128 worlds in all three modes, including commanded arm/jaw
motion. The final 32-world capture matches the saved stage-4 implementation
byte for byte over 23,040 transitions and 32,268 contact manifolds. A separate
65-world capture checks partial final blocks and also matches byte for byte
across 46,800 transitions and 65,521 manifolds. Both have zero nonfinite outputs.
The respective SHA-256 values remain:

- `1f0936dfad2c21ba87a949a8e275ca6f47ec32143e7ce89966253191cb69b3b9`
- `b2d0c2f01854f470424cae9a1689b2dc1f0e74c839ea29239e6f2755e37b9fd1`

This establishes sampled parity with stage 4. It does not remove the documented
stage-3/stage-4 trajectory differences. GPU register use remains 255 per thread;
local storage is 3,760 bytes (previously 3,744), and environment allocation
remains 102,944 bytes per world.

CUDA Compute Sanitizer memcheck also passes the 23,040-transition production
capture with zero errors and the same saved bytes. A source scan finds no
lines over 100 columns or indentation outside four-space multiples.

### Performance check

RTX 3080 Ti, 4,096 worlds, 32-thread blocks, 20 warmup calls, median of five
100-call samples, production hold-open stepping without policy inference:

| Mode | Stage-4 baseline steps/s | Final cleanup steps/s | Throughput change |
| --- | ---: | ---: | ---: |
| Picking | 791,241 | 780,715 | -1.3% |
| Stacking | 400,852 | 411,391 | +2.6% |
| Basketball | 1,256,092 | 1,287,789 | +2.5% |

The final-layout run precedes its baseline comparison. These small mixed
changes are comparable to run-to-run timing variation; this is a code-reduction
stage, not a measured throughput improvement. A rejected smaller-layout trial
measured picking/stacking slower in both before/after and reversed-order
comparisons, so the original world layout is retained. No GPU launch, solver
iteration count, scratch/cache capacity or arithmetic policy is changed.


## Rendering checkpoint

The renderer now submits the same mesh transforms to a 4096-square depth map
and the shaded scene. Articulated GLTF geometry, task objects, the table and
basketball hoop cast and receive directional shadows. A weighted 5-by-5 PCF
filter and receiver-plane depth correction soften edges and suppress acne.
Lighting uses roughness-dependent specular response, sky ambient illumination,
tone mapping and a subtle antialiased table grid.

Rendering adds 69 environment lines relative to stage 5. The fork build-script
compatibility declaration adds seven more, bringing the environment to 3,200 lines. The four GLSL files total
120 lines, up from 51; shader assets are excluded from the environment ledger.
No simulation kernels or physics settings change in this rendering checkpoint.

`tests/test_robot_arm_render.cu` passes picking, stacking and basketball with
moving geometry and no OpenGL errors, including renderer teardown. The full
robot_arm trainer builds, and `./puffer eval` renders successfully during a
15-second virtual-display smoke test. Screenshots were inspected in all three
modes. These rendering checks use Mesa llvmpipe, so they do not establish
hardware graphics throughput.

The shared `puffer` executable had previously been compiled for breakout,
which triggered the zero-agent assertion before robot_arm rendering started.
It is now replaced with the robot_arm build; the previous binary is saved at
`/tmp/ra-shadows/puffer-breakout-backup`. To rebuild for this environment, run
`bash ./build.sh robot_arm`, then `./puffer eval`.
