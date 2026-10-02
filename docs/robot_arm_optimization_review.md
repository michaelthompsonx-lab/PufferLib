# Robot arm optimization and reduction review

Reviewed 2026-10-02 after migration stage 3. This is a source review;
no runtime speedups were measured and no production code was changed at review time.
The first four recommended groups and GPU launch tuning are implemented in
migration stage 4; see
[the migration ledger](robot_arm_puffysics_migration.md) for current counts and
validation. Stage 5 applies `SKILL_ISSUES.md`, consolidates the adapters and
removes 95 further lines while retaining byte-identical stage-4 captures.
GPU storage/broad-phase work remains a profiling follow-up.
At review time, environment source was 3,379 lines, including both adapters,
with a cumulative reduction of 2,174 lines (39.2%). The historical counts and
recommendations below describe that checkpoint; the current source is 3,124
lines, a cumulative reduction of 2,429 lines (43.7%).

| Source | LOC |
| --- | ---: |
| robot_arm.h | 939 |
| robot_arm_cuda.cuh | 899 |
| robot_arm_task.h | 748 |
| robot_arm_render.h | 357 |
| robot_arm_model.cuh | 154 |
| robot_arm.cu | 132 |
| robot_arm_physics.cuh | 124 |
| robot_arm_dynamics.cuh | 26 |

## Recommended order

1. Share FK in physics preparation; consolidate contact telemetry and body classification.
2. Replace the remaining clearance SAT loop with a reusable engine overlap query.
3. Extract explicit force/integration and compliance calculations into reusable engine helpers.
4. Consolidate model and collision-geometry data.
5. Profile GPU memory use and narrow-phase work before changing storage or broad-phase policy.

LOC estimates below are rough net environment savings including new adapters
and model-data headers, not commitments. They overlap and must not be added
as a guaranteed total. Moving environment code between files saves no LOC.

## Share forward kinematics and prepared geometry

`ra_prep()` calls `ra_massm()`, `ra_gravt()` and then `ra_fk()` for body setup.
Both wrappers invoke `ra_dpose()`, which calls FK independently. All three
use the same joint positions and jaw width. Prepare links, origins, axes and
dynamics-body poses once and pass them to the existing Puffysics mass/gravity
APIs. Keep separate mass and gravity accumulation order to preserve rounding.
The mass/gravity wrappers may remain for the native comparison fixture.

A normal substep currently runs FK five times: two dynamics preparations,
body preparation, the candidate table guard, and final post-integration poses.
Eight substeps plus begin and observation produce 42 explicit FK calls on the
normal path. Sharing the three preparation calls removes 16 per control step.
Table-guard bisection adds eight more evaluations in each affected substep.
This is a count of source call paths, not measured GPU execution time.

Further reuse needs care: the table guard evaluates candidate arm coordinates
before jaw-width integration; final poses use the updated width. Observation
needs final joint origins and axes, whereas staged axes currently describe
pre-integration coordinates. Do not reuse those stale values. Audit the begin
FK separately before removing it. Expected LOC benefit is small; this is the
highest-priority runtime candidate.

## Consolidate contact telemetry and classification

`ra_colc()` and the active-pad loop in `ra_solve()` both traverse solved
manifolds and points. A shared pass can collect pad impulse totals, wrist
impulses and active-pad flags. Preserve accumulation order and existing point
thresholds. Wrist moments currently use the post-integration end-effector;
the merged pass must remain after final FK.

Pad-side/body-kind classification is repeated in reaction setup, telemetry,
grasp detection and object filtering. Small helpers can make the ranges and
last influencing joint explicit. `ra_botc()` repeats the same two cube pad
queries in both basketball and other-mode branches; call them once and retain
the stacking-only base queries. Rough net saving: 15–35 lines across these
changes, with modest traversal/branch savings.

## Remaining clearance SAT implementation

`ra_padhit()` occupies 98 lines in robot_arm.h and implements its own 15-axis
box overlap loop. It is used by `ra_stept()` for gripper clearance against the
cube and stacking base. It is live task code, not obsolete contact code.

Extract a generic tolerance-aware OBB overlap primitive into Puffysics, or
reuse an existing query only after proving equivalent margin and degeneracy
semantics. Retain the inward-side filter and face separation checks.
The current routine also constructs/averages witness points, but its two task
call sites only consume the boolean result. Audit external callers before
removing the output-producing path. A boolean clearance API can avoid that
work and keep task-specific clearance decisions local.

Do not substitute compound solver contacts directly: their candidate clipping,
feature selection and hidden-surface rules differ from this task predicate.
Estimated net environment saving: 40–75 lines. Add focused tests around seams,
inward/outward sides, nearly parallel axes and margin boundaries, and compare
reward/termination outputs as well as physics state.

## Reusable explicit dynamics and contact material helpers

The remaining generic pieces include explicit clamped PD force, velocity
updates, joint-limit integration, box/sphere inertia, rigid impulse math,
constant-gravity linear drag, and spring/damper-to-CFM/ERP conversion.
They belong in reusable Puffysics APIs with caller-supplied coefficients,
limits and traits. Keep table-guard policy, object pair selection and task
reward calculations in the environment.

Explicit motors must retain force clamping before acceleration, the present
energy accumulation order, the -qd arm damping term, and velocity limits.
Joint-limit response currently applies a 0.15 multiplier; make that caller
policy rather than baking it into the engine. The jaw uses separate full-width
effective mass. Preserve that coordinate convention.

`ra_react()` counts matching patch-group points once per point. Count group
membership once per manifold, then reuse it for the material calculation.
Manifolds have at most four points, so this is a small optimization. A reusable
compliance helper can remove the material arithmetic without moving robot pad
constants into the engine. Estimated net environment saving for this group:
40–90 lines, depending on the API and retained compatibility wrappers.

## Consolidate model and collision-geometry data

robot_arm.h and robot_arm_model.cuh duplicate ten-body masses, centers and
inertias, seven joint offsets/rest rotations, joint limits and home positions.
A shared robot-specific data source avoids drift between production and the
native fixture. Keep quaternion-layout conversion explicit; native tensors
still undergo principal-axis conversion. Count the new shared header in the
LOC ledger. Estimated net saving: 20–40 lines with compact shared tables.

`ra_padsh()` and `ra_gripb()` branch over fixed geometry descriptions. Tables of
local poses/extents can reduce those branches and centralize definitions used
by body setup and the table guard. Preserve invalid-index behavior if callers
can rely on it. Tables can increase GPU loads, so reduced source length is not
proof of increased speed. Estimated net saving: 15–30 lines.

## GPU storage and collision work: profile first

`ra_bodies()` rebuilds static table/rim/backboard bodies and shapes each
substep. Initialize static geometry on reset/topology changes and update only
moving objects and articulated proxies. Preserve clearing of manifold counts
and component masks every substep, topology-dependent cache invalidation,
and updates of object inertia when mode changes.

Each environment carries 48 manifold slots, 192 cache entries and 128 compound
candidate scratch entries. These are stronger memory/throughput candidates
than removing small helpers, but first measure sizeof(Env), traffic, register
usage, spills and occupancy with the production kernels. Do not reduce
capacities from observed average counts; establish safe bounds and retain
explicit overflow behavior. Shared per-block scratch must be sized for all
active threads; current one-thread-per-world storage cannot simply be reused
concurrently.

`ra_objr()` has a sphere-distance broad phase for links, while shell and pad
queries have separate paths. Test conservative rejection before expensive
compound/narrow-phase queries, using swept bounds where speculative contacts
need them. Preserve pair ordering and feature/cache behavior. These changes
may improve speed without reducing LOC and can increase code length.

## Small cleanup and task/render scope

`RaCollisionBox::pad_face` is assigned but never read in the environment.
`PL_IMPULSE_MAX_CANDIDATES` and `PL_SAT_MAX_CLIP_VERTICES` have no environment
uses beyond their definitions. Remove only after checking external fixture
references. body_count and shape_count currently receive identical updates;
consolidating their bookkeeping is another small candidate. Expect fewer than
15 net lines for this cleanup, rather than a major migration stage.

Basketball trajectory quality calls `ra_bxing()`; release statistics and the
renderer can request the same crossing again for the same position/velocity.
Return crossing data with trajectory quality and reuse it within each call
site. Move general gravity/drag prediction math to Puffysics, while keeping
hoop dimensions, quality weighting and scoring thresholds in the task.

The 748-line task file contains observations, reset, episode statistics and
mode-specific reward state machines. Most is environment behavior. Keep reward
coefficients, transition order, observation layout and reset RNG sequence
stable; aggressive unification can hide meaningful mode differences.
Rendering is 357 lines and does not run in the training physics kernels.
Rendering cleanup is therefore lower priority for training throughput.

## Verification for implementation stages

Use the existing independent engine tests and 23,040-transition production
capture for behavioral changes. Add focused boundary fixtures for any new
clearance or limit API; rollout parity alone does not exercise every branch.
Use tools/bench_robot_arm.cu for a comparable production baseline, extending
its hold-open workload with scripted contact-heavy and table-guard scenarios
before claiming a broad speedup. Record mode, world count, GPU, timing method
and median repeated results. Keep runtime gains and net LOC reduction as
separate measurements.
