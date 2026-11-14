# H20 deployment record and 64/128-GPU launchers

This directory versions the actual deployment adaptations, preparation and
validation tools, ownership fixes, and an automatic 128-GPU job entry point.
The platform job must explicitly select the published Git commit and GPU count.
The previous 64-GPU allocation is separate from a newly submitted 128-GPU job.

## Source history and validation scope

- Base: upstream PEAR `f32aa1f7c2ec9c3f45630cc9f8541b3201e8e087`.
- `d96355d` records the five source files read back from the deployed source:
  local runtime paths, CUDA compatibility library selection, headless LIBERO
  installation, the H20 LMCache native allocator selection, and GDN arithmetic
  operation ordering. It does not change the GRPO objective or reward function.
- `1c1cc81` captures the then-deployed 64-GPU launcher and its helper scripts.
  `provenance/deployed64_sha256.json` identifies that original launcher.
  The downloader's corporate proxy was replaced with environment configuration
  when publishing the tools.
- Later launcher changes are independently reviewable against that snapshot.
  The preparation and validation helpers in `runtime/` preserve the deployment
  workflow; they require the existing model, runtime bundle and manifests.
  They are not a portable, one-command environment installer.
- `provenance/*.sh.snapshot` are historical **32-node** launch scripts, retained
  only as evidence. They are not active entry points. The active launcher here
  supports 64 GPUs (eight nodes, two trainer/six rollout) and 128 GPUs
  (sixteen nodes, four trainer/twelve rollout), selected by `PEAR_TOTAL_GPUS`.
- The upstream veRL and LMCache patch series are already in `3rdparty/`; this
  change adds no new commits to those dependency repositories or internal SCM
  repositories. Runtime archives, model weights, checkpoints, authentication
  tokens and raw platform configuration are not included.

The original source/runtime completed one functional **8-GPU GRPO step** and
wrote a checkpoint. Its rewards, policy loss and gradient norm were zero, so
this establishes execution, not learning effectiveness. Historical kernel
validation had 24 passes, one recorded FLA FP64-reference difference and four
Blackwell-only skips; 458 compatibility checks passed. The eight-node collective
check used one GPU per node and passed NCCL/Gloo; **64-GPU training did not
complete a step** before the user paused it. Sanitized summaries are in
`provenance/`. None of this certifies the new launcher on GPUs.

## Launch ownership and cleanup

`merlin_stage1_entry.sh` is the public entry point. It invokes
`launcher_control.py` before network selection, runtime restoration, or failure
publication; `worker_entry.sh` and `stage1_node.sh` are internal children.

1. A permanent `launches/<id>/ranks/<rank>` claim owns that invocation. A repeated
   rank exits with status 75 and cannot overwrite the healthy launch's failure
   record. Reusing old-format rank/publisher claims is also rejected.
2. Rank zero claims the canonical RUN directory with an atomic `.pear-active`
   directory. Different launch IDs cannot write the same RUN concurrently;
   symlink aliases resolve to the same lock. Every worker validates the published
   RUN owner. A node-local lock also excludes overlapping launches on one node.
3. The Linux child subreaper supervises the complete owned process tree. Cleanup
   continues after the original leader exits and includes detached/double-forked
   children. Signals use PID, UID and process start-time identity; unrelated
   processes are not selected by command name, environment tag or GPU use.
4. Node ownership is released only after local cleanup. Rank zero releases RUN
   only after all ranks acknowledge cleanup against the same owner nonce.
   Missing workers or uncertain cleanup leave the lock in place. Retired lock
   directories remain as audit records; there is no automatic stale-lock TTL.

These locks assume the shared filesystem supports atomic directory creation and
rename. RUN and `PEAR_ROOT/launches` must be on the same shared filesystem.
Do not manually remove a retained lock until all processes for its recorded
owner have been checked and stopped on every worker. A new launch ID alone does
not override a retained RUN/node lock.

## Job entry point and configuration

The platform entry invokes `job_entrypoint.sh` from an immutable NAS release.
It verifies every file against `release.json` and the configured commit, starts
the owning launcher once, records its exit status, and finishes with
`exec sleep infinity`. Both successful completion and training failure retain
the allocation for inspection; the platform's `running` state alone therefore
does not prove training is active. There is no automatic retry or lock removal.
An outer `sleep infinity` in the job configuration also covers bootstrap failures.

Supply the same `PEAR_ROOT`, `PEAR_TOTAL_GPUS`, `PEAR_DEPLOY_COMMIT`,
`PEAR_LAUNCH_ID`, `PEAR_RUN` (if set), and `PEAR_RESUME` on all worker pods.
Merlin supplies `ARNOLD_ROLE=worker`, `ARNOLD_ID=0..15` for 128 GPUs and
`ARNOLD_TRIAL_ID`. The new 128-GPU job needs no separate CPU head.

The platform wrapper defaults to 128 GPUs; the lower-level direct launcher
defaults to 64 for compatibility. Production job configuration sets 128 explicitly.

| Variable | Meaning |
| --- | --- |
| `PEAR_TOTAL_GPUS` | `64` or `128`; topology, barriers and collective validation use the same selector |
| `PEAR_DEPLOY_COMMIT` | Exact 40-character Git commit of the published NAS release |
| `PEAR_ROOT` | Existing shared deployment root; defaults to the recorded NAS root in the scripts |
| `PEAR_LAUNCH_ID` | Unique attempt ID; the platform wrapper uses `merlin-trial-${ARNOLD_TRIAL_ID}-stage1-${PEAR_TOTAL_GPUS}-v1` |
| `PEAR_RUN` | Defaults to `${PEAR_ROOT}/work_dir/runs/${PEAR_LAUNCH_ID}` |
| `PEAR_RESUME` | `0` by default; `1` requires an explicit RUN and a new launch ID |
| `PEAR_NODE_LOCK` | Defaults to `/var/tmp/pear-merlin-$(id -u).active`; keep identical across attempts on a node |
| `PEAR_NETWORK_INTERFACE` | Node-local interface, defaults to `eth1`; propagated consistently to Ray/NCCL/Gloo validation |

For an existing nonempty RUN, explicit resume only permits opening that output
directory; the training framework still validates checkpoint compatibility.
The launcher does not perform GPU scheduling or choose a resource queue.

The existing NAS `$PEAR_ROOT/launch` is a **frozen validation input**, including
`restore_runtime.sh`, `preflight_stage1.py` and the original helper hashes.
Run the launcher from its separate fixed-commit release. Training uses the frozen
`$PEAR_ROOT/repo` tree whose original 837-file hash must match the recorded source
identity; no training-source change is introduced by the topology/entrypoint update. Do not overwrite the frozen
helpers or rewrite the old certificate to make it cover this new launcher.
The original 837-file source manifest and the independently recorded launcher
hashes intentionally have separate scopes.

## CPU regression tests

On Linux with Python 3.10+ and `/proc` access:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s deploy/merlin/tests -p 'test_*.py' -v
```

The tests use bounded temporary file/process fixtures and do not import Ray,
PyTorch or CUDA, start training, or submit a task. They cover duplicate claims,
RUN contention/resume, cleanup acknowledgements, orphaned and detached children,
signal delivery, and preservation of unrelated processes. Non-Linux runs skip
these tests and must not be reported as Linux validation.

## Progress and Tracking

The controller writes failure/cleanup records under
`$PEAR_ROOT/launches/$PEAR_LAUNCH_ID`. Per-node bootstrap logs and final exit
status are under `$PEAR_ROOT/entrypoints/$PEAR_LAUNCH_ID`. `logs/training.log`
and `$PEAR_RUN/train.log` contain driver output; `progress.json` contains
trajectory/group counts and the trainer phase; `metrics_*.jsonl` and
`tensorboard/` contain training metrics; `ckpt/` contains checkpoints.

The image's separate system Python runs `tracking_bridge.py` on rank zero. It
uses byted-wandb to mirror real JSONL metrics to the internal Tracking service,
without changing the validated training environment's official wandb package.
The true SDK URL is recorded in `tracking.json`; no training Run is fabricated
while a job is waiting in the scheduler. See `TRACKING.md` for bridge details.
