# Internal Tracking for PEAR

The certified training environment contains official `wandb==0.31.0`. Do not replace it or enable its external W&B backend for this deployment. The image Python has **byted-wandb 0.13.98**, which the independent CPU bridge uses. The training recipe keeps `LOGGER='["console","file","tensorboard"]'`; no training/model/kernel changes are needed.

`tracking_bridge.py` reads the original `RUN/metrics_*.jsonl` records (`{"step": n, "data": {...}}`), retaining their metric names, numeric values and training step. It does not generate loss, reward, throughput or progress points. It never creates or writes files in RUN. Its SDK directory, persistent state, claim and `tracking.json` are under CONTROL. Controller ownership must already have created RUN before it starts.

## Start and stop

Run one bridge on rank zero using the **image Python captured before activating the training environment**, not `VERL_ENV/bin/python`:

```bash
"$PEAR_TRACKING_PYTHON" "$LAUNCH/tracking_bridge.py" \
  --run "$RUN" --control "$CONTROL" --launch-id "$LAUNCH_ID" \
  --project pear-h20 > "$CONTROL/logs/tracking.log" 2>&1 &
TRACKING_PID=$!
```

The stage wrapper must treat this as an independent optional logger: a bridge failure must not change the training command's exit status or terminate training. Never start it before the controller has acquired RUN, because SDK log directories would otherwise make a fresh RUN appear nonempty.

After the real training command exits, atomically publish `CONTROL/tracking.stop.json` with `{"exit_code": <actual training exit code>}`. The bridge reads any final complete records, flushes its buffered last step, calls SDK `finish(exit_code=...)`, writes its final status, and releases only its own claim. Wait at most 35 seconds in the wrapper: SDK finish has a 30-second deadline. SIGTERM/SIGINT are fallback stop requests; they also attempt final flush/finish and record status 143/130. Forced process death can still prevent final SDK flush.

The bridge cannot signal the training process. If it crashes, a new invocation with the same CONTROL resumes its persisted offsets and original SDK unique ID after a normal claim release. A SIGKILL can leave `tracking-bridge.claim`; it is deliberately not stolen automatically. First verify the old bridge is gone before an operator archives that claim.

For a **new empty RUN**, `--metrics-since 0` (the default) is correct. For a new launch resuming an existing RUN, choose explicitly:

- Preserve the whole run's metric history in the new Tracking run: keep zero; historical JSONL points will be replayed with their original steps, not relabelled as new work.
- Show only the new launch: pass the launch start epoch via `--metrics-since`, captured before training creates its new metrics file. Discovery excludes older files by mtime. Preserve the same value if restarting this bridge's CONTROL; it is part of its persistent source identity.

## Merge, durability and limitations

Only newline-terminated JSONL records are consumed. Each source offset and pending metrics dictionary are saved together atomically. Records for the same step are merged before sending; for repeated keys, the last complete record wins. Earlier steps flush when a newer step is seen; the newest step flushes after two seconds without a new record for that step, or immediately during shutdown. The actual SDK call is one `wandb.log(data, step=original_step, commit=True)` per submitted step.

Already submitted steps are not logged again: exact repeats increment `duplicates`; a differing late record increments `late_conflicts` and remains available in the original JSONL. A late record cannot retroactively merge into an already committed step. Ordinary PEAR FileLogger emits the full metric dictionary in one line per step, so this is a defensive boundary for malformed/reordered producers. Non-numeric/non-finite values are not presented as scalar metrics; the state records `non_scalar_values`. Truncation or replacement of a consumed metrics file stops the bridge rather than guessing an offset.

`submitted` means the SDK log call returned, **not** that the remote backend acknowledged durable storage. The SDK is asynchronous. A crash between log and state save can replay a point; a crash after state save but before SDK upload can lose an unsent point. Stable SDK ID/resume and step checks limit duplication, but the bridge does not claim exactly-once remote delivery. Retain the original JSONL as the source of truth and compare actual platform metrics before calling upload verified. `sdk_finish_completed` records successful final flushing as reported by the SDK.

## Required identity and real links

Read-only inspection of the exact 0.13.98 wheel confirmed:

- `wandb/sdk/internal/tracking_cloud.py:119-148`: `RH2_JOB_RUN_ID` gives `outer_type="merlin"` and the linked job ID. `SEED_INSTANCE_ID` and `SEED_ARENA_EVALUATION_ID` take precedence; the bridge refuses these conflicting evaluation identities.
- Same file, lines 151-179: `ARNOLD_TRIAL_ID` and `ROBUST_RUN_ID` populate outer_info; `RH2_HOST` containing `seed` sets is_seed. Keep the platform's real environment; do not synthesize another task's IDs.
- `wandb/sdk/wandb_run.py:997-1006`: `run.get_url()` reads `TrackingParameter.run_link`, which originates in the actual create-run response. The bridge records this returned URL and extracts its actual `Id`/`selectedTrial`; it never constructs a run URL from a desired name.
- `tracking_client.py:create_project` and `tracking_internal_api.py:upsert_run`: SDK project creation uses its internal SDK endpoint with a project name; no entity/namespace parameter is required. The public CLI schema also requires only name.

The bridge rejects official W&B mode, offline/disabled settings and an unreviewed byted-wandb version. It requires the real `RH2_JOB_RUN_ID` and numeric `ARNOLD_TRIAL_ID`. It uses project **pear-h20** and run name **PEAR_LAUNCH_ID**, with a short stable SDK unique ID derived from the source identity. Authentication/network settings otherwise come from the image's platform environment. Never set `WANDB_OFFICIAL=1` or request an external W&B API key.

Read-only package/identity probe (does not initialize a run):

```python
import importlib.metadata, importlib.util, os
print(importlib.metadata.version("byted-wandb"))
print(importlib.util.find_spec("wandb").origin)
for name in ("RH2_JOB_RUN_ID", "ARNOLD_TRIAL_ID", "ROBUST_RUN_ID", "RH2_HOST",
             "SEED_INSTANCE_ID", "SEED_ARENA_EVALUATION_ID"):
    print(name, os.environ.get(name))
```

`CONTROL/tracking.json` is the local discovery/status record. A real project may exist while a 128-GPU job queues; an empty project is not a training Run or a successful training step. SDK initialization creates a real Run only when the bridge starts in the actual job. Its first model metrics appear only after PEAR logs its first real training step.

## Platform verification

The project was independently read back on 2026-11-15: **pear-h20**, `project_20261115_9a949690`. The actual returned project URL is:

<https://ml.bytedance.net/experiment/tracking/detail?Id=project_20261115_9a949690>

The initial project-create request returned HTTP 404 even though its side effect succeeded; subsequent exact `projects get` and `MyCreated` list confirmed the project. Do not retry a write solely because its response failed. No PEAR run/metrics were created by the CPU tests or by this source review.

Query actual Runs associated with the new job; match the returned `outer_info.arnold_trial_id` and name before sharing `url`:

```bash
merlin-cli tracking runs list --json '{
  "filter":{"outer_ref":{"type":"merlin","id":"ACTUAL_JOB_ID"}},
  "options":{"include_url":true,"include_outer_url":true,"include_latest_logged_at":true},
  "page_size":100
}'
```

Use the returned project/run/region in `tracking run-entities list` to discover real metric names. Scalar API casing differs from entity API:

```bash
merlin-cli tracking run-scalar-charts get --json '{
  "ProjectRunIds":[{"ProjectId":"ACTUAL_PROJECT_ID","RunId":"ACTUAL_RUN_ID","Region":"cn"}],
  "XAxis":{"Name":"Step"},
  "YAxis":{"Metrics":[{"Name":"ACTUAL_METRIC_NAME"}]},
  "Option":{"DisableDownsample":true}
}'
```

Cross-check the returned step/value against the original JSONL. `latest_logged_at` is preferable to metadata `updated_at` for checking whether points are arriving. A project or Run link alone is not proof that training metrics have arrived.

## CPU regression coverage

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s deploy/merlin/tests -p test_tracking_bridge.py -v
```

These tests use an explicitly substituted recording backend; they never import/init the real SDK or create a platform Run. Coverage includes incomplete lines, same-step merging, last-step flush, persistent recovery/deduplication, late conflicts, failed submit, old-file filtering, RUN read-only behavior, exclusive claims, real subprocess TERM handling, normal stop-file handling, and identity/link parsing. Actual SDK authentication and remote delivery must still be verified in the real job.

Upstream logging evidence: `examples/models/qwen3_8_27b_pear/train/run_stage1.sh:39-40,93,112`; `pear/configs/base.yaml:23,114-116`; installed veRL `verl/utils/tracking.py:76-87,530-568`; `verl/experimental/separation/ray_trainer.py:293-298,763`; `pear/trainer/runner.py:54-58`. The original trainer already batches one metrics dictionary per step and explicitly finishes its configured loggers.
