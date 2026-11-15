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

## Image Python path portability

The job entry resolves the requested image Python before environment activation.
If a copied job specifies a missing executable (for example `/usr/local/bin/python3`
in an image that provides `/usr/bin/python3`), it falls back to `python3` on PATH
and exports the resolved executable to the independent Tracking bridge. This
also ensures release verification and final status recording use a working
interpreter. New job configurations should omit a hardcoded interpreter override.

## Ray descriptor capacity and large-cache shutdown

The owning controller raises the inherited open-file soft limit to 262144
(`PEAR_NOFILE`), capped by the hard limit, before spawning any worker or Ray
process. It preserves higher existing limits and refuses a hard limit below
65536. `CONTROL/limits.<rank>.json` records before/after values. The earlier
1024 soft limit exhausted GCS sockets as hundreds of actors registered.
Kernel FD table presizing in `sitecustomize` is separate from raising this limit.

The controller allows 120 seconds after SIGKILL for large pinned-memory workers
to finish kernel teardown and be reaped, then retains ownership if any child
remains. Rank zero waits up to 300 seconds for all cleanup acknowledgements.
No grace period permits unsafe lock takeover.

## Worker-group master port allocation

`pear_merlin_ports` is loaded through veRL's external-module hook for this
deployment. It replaces only the worker-group master-address allocator. Ports
are selected from 21000–29999, validated against each node's actual ephemeral
range, and claimed atomically in a per-launch node-local directory. A claim
persists after the probe socket closes, preventing another group from selecting
the same port while rank zero is still being created. Existing listeners are
skipped. No kernel/network sysctl, training objective, model or runtime package
is changed. Ray's worker port range remains 3000–5999. The old EADDRINUSE at
TCPStore startup was observed with a master port in the ephemeral range.

The launcher propagates the plugin, its fixed-release Python path and the same
launch ID through Ray runtime_env. CPU regressions exercise real concurrent
processes and delayed socket binding; a separate actual-runtime check loads the
veRL external hook and creates two real CPU TCPStores with successful round trips.


## Storage outage and explicit live recovery

The supervisor keeps a private authoritative journal under `/var/tmp/pear-owner-*`.
Its shared state includes `local_state_path`. NAS mirroring errors are recorded
locally and reported without abandoning the process tree. Even failure of both
recording locations cannot bypass identity-checked cleanup signals; inability
to persist final cleanup evidence still retains the launch locks.

`recover_owned_launch.py` is an explicit operator recovery tool for a launch
whose original supervisor died. It refuses a live original supervisor, a stop
already requested, another host/rank/UID, or a changed PID birth identity. It
starts no GPU training command. Existing processes are monitored using the
original identity records and invocation markers; new CPU helpers are children
of a new subreaper. Existing PID-1 orphans cannot be reparented for kernel
reaping, so the tool never certifies original launch cleanup or releases the
original RUN/node locks. Those locks require a later operator audit.

With `--outputs <HDFS-mount-directory>`, recovery records, progress output,
Tracking state and new diagnostic logs go to HDFS. Tracking resumes the same
SDK ID and reads the original unmodified metric stream. Its SDK binary queue
uses private local scratch. Checkpoints use an HDFS directory when the current
checkpoint directory is absent or empty. Existing open legacy log/metric file
descriptors cannot be rerouted by replacing a pathname: they remain live and
are mirrored byte-for-byte to HDFS. The tool does not silently claim these old
file descriptors stopped writing NAS. This preserves live sampling instead of
restarting GPU actors merely to relocate their descriptors.

`reroute_agent_logs.py` changes only diagnostic references on authenticated
existing async agent-loop actors. It verifies trial, launch, actor class and
filenames first, uses per-host directories to avoid PID collisions, and leaves
policy parameters, rewards, sampling and episode state untouched.


### Recovered Tracking run still labeled crashed

The reviewed byted-wandb 0.13.98 client can resume streaming into the original
Run without clearing its old `crashed` metadata label. This is separate from
server acknowledgement of scalar points. `repair_tracking_status.py` is an
explicit operator recovery command; it requires fresh streaming and guardian
journals, live original-root and producer birth identities, and exact
project/run/job/trial association. It only refreshes the heartbeat and changes a
`crashed` label to `running`, then reads the server back. It cannot revive a
finished/failed run, initialize a second producer, or write scalar points.

Use the image Python, the actual recovered HDFS control directory, and its
rank-0 recovery directory. Save the repair utility in a new fixed-commit HDFS
release; do not edit an existing release or restart the GPU training to fix a
Tracking label. The utility writes its before/after audit to HDFS
`control/tracking_status_repair.json`. Its seven CPU tests cover delayed server readback, wrong association,
failed lookups, terminal states and SDK errors swallowed before readback.
