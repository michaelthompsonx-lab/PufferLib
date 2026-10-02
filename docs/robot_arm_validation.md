# Robot arm native migration: validation and performance

Date: 2026-09-26. GPU: NVIDIA GeForce RTX 3080 Ti, driver 595.71.05.
Build: CUDA, C++17, `-O3 -lineinfo -arch=sm_86`, float physics.

Latest implementation results: [profile-guided optimizations](puffysics_profile_optimizations.md), 2026-09-27. Retained 64-thread native fixture runs measured 504–531k picking, 298–317k stacking and 815–869k basketball control steps/s at 4096 worlds. Earlier measurements below are retained for provenance.

Production update, 2026-10-02: margin-aware SAT and the persistent impulse
solver now live in Puffysics, with production calling them through an adapter.
The second stage also extracts compound contact generation and swept queries,
with byte-for-byte comparisons of sampled manifolds, patch data and ring sweeps.
The third stage extracts serial-chain mass, gravity, linear solves, contact
responses and body velocity calculations into Puffysics. Explicit motor and
separate jaw updates remain in production. Stage 4 shares FK preparation,
extracts explicit update and clearance helpers, and consolidates robot data.
Stages 1–3 match the previous scripted capture byte for byte. Stage 4 preserves
sampled categorical state and terminals, with floating-point trajectory drift;
see [migration details](robot_arm_puffysics_migration.md). Its smaller GPU launch
layout matches the final implementation byte for byte across block sizes. This is
separate from adopting the native articulation described below. Stage 5 applies
the repository style/reduction guide, removes 95 further environment lines,
and preserves the stage-4 captures byte for byte across 69,840 transitions.
The native fixture now calls the shared reference dynamics APIs directly; its
comparison coverage and tolerances are unchanged.

## Scope

The native robot model is still articulation-only. Production `robot_arm`
uses extracted Puffysics contact and serial-chain dynamics APIs with its
existing model, FK, motor updates and integration. These results validate the new model
and coupling implementation; they do not establish a completed migration,
MuJoCo parity, task success, or training performance.

The native harness adds an infinite floor and object geometry solely as a
validation fixture. There are no robot collision geometries, finger-pad
contacts, hoop, backboard, production task logic or ball drag in that fixture.
The production baseline uses the existing begin/physics/finish kernels with
zero arm actions and an open gripper. Neither benchmark includes policy
inference, rendering, CUDA graph capture or trainer copies.

## Correctness checks

`tests/test_robot_arm_native.cu` runs actual device functions. It checks:

- Construction of picking, stacking and basketball models.
- Rejection of cyclic/zero-scale mimic declarations and invalid/duplicate
  explicit collision pairs.
- Version-002 model cache round-trip, retaining coupling and pair policy.
- FK against the existing robot's body poses over 128 varied configurations.
- The seven-arm-coordinate mass block and gravity forces against the existing
  full-tensor implementation. This also checks the principal-frame conversion.
- Central finite differences of all ten robot body-origin linear Jacobians
  against the eight independent robot coordinates.
- Gripper half-width mass of 0.23 kg, and exactly one coordinate integration
  despite two finger links.
- Negative follower ratio and offset, corresponding velocity, and motor
  reaction scaling by virtual work.
- Exact state save/restore, including controls, applied forces and time.
- A stationary native floor fixture and commanded arm/jaw motion.

Initial 128-world comparison maxima (same robot data in all three modes):

| Quantity | Maximum absolute error | Acceptance limit |
| --- | ---: | ---: |
| Body position / rotated unit-axis difference | 1.674e-6 | 3e-6 |
| Arm mass-matrix entry | 4.649e-6 | 1e-5 |
| Gravity torque | 1.717e-5 | 1e-4 |
| Body-origin linear Jacobian, finite difference | 4.241e-4 | 1e-3 |
| Half-width mass | 4.470e-8 kg | 1e-6 kg |
| Single integration of half-width | 4.657e-10 m | 1e-8 m |

The final stationary run completed 4,960 substeps (10.333 simulated seconds)
in every mode with no failed world or nonfinite state. Largest final arm
position error from its home target was 0.006294 rad; final speed was below
6e-5 in generalized velocity units. Width stayed at 0.08 m.

Each mode then ran a further 1.5 simulated seconds with joint targets shifted
by +0.2, -0.1 and +0.1 rad on joints 1, 2 and 6, and width commanded to 0.02 m.
All 128 worlds per mode passed the motion checks. Final generalized speed was
below 6e-5; measured width-target error was below 4e-9 m. The harness checks
the maximum arm displacement is between 0.19 and 0.21 rad; it does not assert
zero gravity-dependent error on each individual joint.

CUDA Compute Sanitizer memcheck completed the quick suite with **0 errors**:
eight worlds per mode, coupling/reference/cache/state checks and 0.667 simulated
seconds of stationary floor contact. The quick suite skips the longer commanded
motion phase. Instrumented timings are excluded from the benchmark table.

Cube center height settled to approximately 0.031077 m (half-height 0.035 m),
and sphere center height to 0.024076 m (radius 0.028 m). The approximately
3.924 mm penetration is consistent with the fixture's compliant material:
`9.81 * 0.02^2 = 0.003924`. It is not a rigid-contact accuracy result and should
not be adopted as a robot-pad or stacking tolerance.

## Performance

4,096 worlds, eight physics substeps per 1/60-second control step. CUDA events
on a non-default stream; 20 warm-up control calls, five measurements of 20
control calls each. Times below are medians. No simultaneous benchmark process
was run. The GPU also serves the desktop, so these are workstation measurements.
The native fixture continues across the five samples; the existing environment
is reset to the same seeded initial states and warmed up before each sample.

"Control steps/s" counts independent environment transitions across the batch,
not batches or physics substeps. Multiply by eight for physics substeps/s.

| Mode | Native fixture control steps/s | Native batch control time | Existing environment control steps/s | Existing batch control time |
| --- | ---: | ---: | ---: | ---: |
| Picking | 158,869 | 25.782 ms | 603,382 | 6.788 ms |
| Stacking | 54,391 | 75.306 ms | 309,747 | 13.224 ms |
| Basketball | 229,180 | 17.872 ms | 1,043,283 | 3.926 ms |

Native measurement ranges for 20 calls: picking 509.750–522.437 ms, stacking
1496.432–1511.484 ms, basketball 356.371–360.940 ms. Existing environment ranges:
133.058–143.251 ms, 253.146–265.329 ms, and 74.226–79.750 ms respectively.

Explicit native per-world buffer allocations total 93.19 MiB for picking or
basketball, and 126.59 MiB for stacking, at 96 rows and 32 contacts per world.
The existing environment and I/O buffers total 403.36 MiB, using float-sized
observation storage in this standalone baseline. These are allocation-size
calculations, excluding CUDA context, kernel stacks, model arrays and harness
diagnostic buffers; they are not measured whole-process VRAM peaks. Contact
capacity and represented geometry differ between the two implementations.

**The native path currently has no demonstrated speed advantage.** It is slower
even in this simpler collision fixture. The timings are not an equivalent-scene
speedup comparison, and neither implementation was compared with MuJoCo.

### Optimization leads

Compiler resource output for `pf_native_step_kernel`: 138 registers/thread,
1,728-byte stack frame, zero compiler-reported register spill loads/stores.
This is static compiler information, not a hardware-counter profile.

Source-level candidates, to measure before selecting an optimization:

1. Dense factorization and contact-response solves include all 14/20 velocity
   coordinates. The arm and free objects have independent mass blocks; exploit
   that structure while retaining coupled contact responses.
2. Each CUDA thread handles a whole world with serial loops. At 4,096 worlds
   and 64 threads/block, the launch has only 64 blocks. Assess cooperative
   work per articulation and batching/layout changes for latency hiding.
3. Jacobian, mass and row arrays are arranged per world. Assess access
   coalescing across worlds and avoid repeated work on structurally zero entries.
4. Reassess capacities and kernel stack use once the real contact model is
   available. Do not tune away required contacts to improve a benchmark.

## Reproduction

From the repository root:

```sh
nvcc -std=c++17 -O3 -lineinfo -arch=sm_86 -cudart shared \
  -I src -I raylib-5.5_linux_amd64/include \
  tests/test_robot_arm_native.cu -o /tmp/test_robot_arm_native

nvcc -std=c++17 -O3 -lineinfo -arch=sm_86 -cudart shared \
  -I src -I raylib-5.5_linux_amd64/include \
  tools/bench_robot_arm.cu -o /tmp/bench_robot_arm

env LD_LIBRARY_PATH=/run/opengl-driver/lib /tmp/test_robot_arm_native
env LD_LIBRARY_PATH=/run/opengl-driver/lib /tmp/test_robot_arm_native --bench 4096 20
env LD_LIBRARY_PATH=/run/opengl-driver/lib /tmp/bench_robot_arm 4096 20
env LD_LIBRARY_PATH=/run/opengl-driver/lib compute-sanitizer \
  --tool memcheck --error-exitcode 99 /tmp/test_robot_arm_native --quick
```

The driver library path is specific to this machine. Sanitizer-instrumented
timings must not be used as performance results.

## Throughput rerun — 2026-09-27

Rebuilt both current harnesses with the flags above. The user stopped two
competing GPU probe jobs before measurement; the compute-process list was
empty before and after the benchmark sequence. Desktop graphics remained
active, and GPU clocks were not locked. Runs were sequential: optimized native,
production, saved pre-optimization native, optimized native again.

Same settings: 4,096 worlds, eight substeps/control step, 20 warm-up control
calls, five samples of 20 calls. Each table entry is a run's median. The
native optimizations skip exactly-zero impulse updates and redundant dense
Jacobian clears. No physics implementation changed during this rerun.

| Mode | Native optimized run 1 steps/s | Native optimized run 2 steps/s | Native before optimizations steps/s | Production steps/s |
|---|---:|---:|---:|---:|
| Picking | 173,415 | 173,736 | 171,521 | 645,208 |
| Stacking | 60,019 | 59,494 | 59,031 | 336,540 |
| Basketball | 252,454 | 272,817 | 249,598 | 1,145,082 |

Steps/s counts independent environment control transitions, not batches.

| Mode | Native run 1 batch ms | Native run 2 batch ms | Native before batch ms | Production batch ms |
|---|---:|---:|---:|---:|
| Picking | 23.620 | 23.576 | 23.880 | 6.348 |
| Stacking | 68.245 | 68.847 | 69.387 | 12.171 |
| Basketball | 16.225 | 15.014 | 16.410 | 3.577 |

Native sample ranges, milliseconds for 20 control calls:

| Mode | Optimized run 1 | Before optimizations | Optimized run 2 |
|---|---:|---:|---:|
| Picking | 461.982–478.036 | 465.329–479.727 | 467.383–475.279 |
| Stacking | 1360.729–1375.429 | 1354.337–1401.550 | 1274.760–1382.663 |
| Basketball | 313.902–326.286 | 325.397–331.080 | 299.689–304.339 |

Picking and stacking throughput improvements over the matched old binary are
only approximately 1–2%, with overlapping sample ranges. Basketball varies
appreciably between repeats. These data do not establish a large or robust
whole-step gain from the two small optimizations. They also explain why the
earlier contended runs must not be used as optimization baselines.

All native runs passed their stationary fixture assertions with identical
printed final state summaries. Every production sample reported zero nonfinite
values. These are throughput-run checks, not a rerun of the full correctness
suite. Native still uses the simpler articulation-plus-floor fixture described
above; production includes a different collision/task workload. The production
numbers therefore remain contextual, not an equivalent-scene physics ratio.
