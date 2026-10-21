# DSN 2026 LID — Team Workflow and Handoff

## Purpose and current status

This document accompanies the team's staged Kaggle notebook. It records the tested workflow and recovery lessons. It is not an official DSN starting kit.

**Latest verified state:** the recovery-run low-LR step-800 candidate achieved a local represented-class diagnostic of **0.5806749508**. Its fp32 export, documentation and supplied model-folder check passed. A model archive was downloaded and extracted locally. Public HF revision verification and competition scoring have not been confirmed in the shared results.

Use the companion model-card README for architecture, labels, training data, limitations and attribution. Do not replace the team's existing root README without agreement; the model card can be stored on GitHub as `MODEL_CARD.md` and used on HF as `README.md`.

## Suggested GitHub contents

```text
DSN-2026-BOOTCAMP-HACKATHON/
├── dsn_lid_kaggle.ipynb
├── LID_WORKFLOW.md
├── MODEL_CARD.md
└── .gitignore
```

**No model weights, dataset audio, private backup ZIPs, tokens or submission artifacts belong in this GitHub handoff.** Keep checkpoints as private inputs/backups and publish only the final permitted files to the competition's HF model repository.

For a registered team, use a branch and pull request to avoid overwriting teammates' work. Check repository visibility before uploading. The competition prohibits sharing submissions between teams; seek organiser clarification before cross-team sharing. Each teammate must use their own authorised credentials, never another person's token.

## Prepare the shared notebook

Work on a copy, not the live notebook executing training:

1. Set `STAGE = "preflight"` as the default.
2. Clear all saved outputs and remove secrets/private authentication material.
3. Remove temporary debugging/download cells and extra standalone `run_stage(...)` calls.
4. Preserve original function definitions and the single main stage dispatcher.
5. Replace account-specific input paths with documented configuration placeholders.
6. Record the package environment actually tested; do not claim cross-version compatibility.

The stage selector controls the main dispatcher, **not arbitrary helper cells**. A helper cell that directly calls export/checking will execute during Run All regardless of the main stage unless guarded or disabled.

## Kaggle prerequisites

- Enable a GPU accelerator and Internet when downloading the official dataset.
- Attach the team's authorised epoch-1 checkpoint as a private input.
- Keep `config.json`, `preprocessor_config.json` and `model.safetensors` together.
- Set `BASELINE_DIR` to the checkpoint folder, not its weights file.
- If access requires authentication, create a Kaggle Secret named `HF_TOKEN` and enable it for this notebook.

```python
"USE_KAGGLE_HF_SECRET": True,
"HF_SECRET_NAME": "HF_TOKEN",
```

`HF_TOKEN` above is the secret name, not its value. Never put the actual value in source or outputs. A copied notebook may need fresh permission to read an existing secret. The HF account and token must have appropriate dataset read access.

The tested workflow uses `cuda:0` only, even when two T4 GPUs are visible. Do not silently change batch size or introduce multi-GPU execution during controlled reproduction.

## Dataset and baseline integrity

Dataset: `Data-Science-Nigeria/dsn-2026-lid-train`.

Pinned revision: `87cf6d4b1c7599f97f44b6e6aade86438369a2b1`.

Audit: 24,774 train / 4,352 validation clips; zero shared IDs and known source/speaker pairs; zero Nupe validation clips. These checks do not rule out all possible audio duplication or missing-metadata overlap.

Original epoch-1 SHA256:

```text
config.json
abac16bf1daa454ebbffaac33e8ecfdacd01f9875ce032a3dafc9bb89c929737

preprocessor_config.json
8cdfd65ff4115423185a1512bdae100e2e0cd744f5b322417429944aaafd0827

model.safetensors
3785f80820ca8bbb2d9fc1d7f43bf7e06856987ea7ccf456f2d7b424165e2a6d
```

Keep the baseline read-only. Never bypass fingerprint or label-order assertions merely to get a stage to run.

## Stage sequence

Edit `STAGE` in the main configuration, then run the original notebook from the top. Do not restart between stages unless needed.

| Stage | Purpose |
|---|---|
| `preflight` | Discover inputs, GPU and disk availability; no training |
| `baseline` | Verify fingerprints, labels, dtype and local loading |
| `pilot` | Optional 64-clip full-audio timing feasibility check |
| `compare` | Save A/B/C comparison; required guard before training |
| `reference` | Save a matching local baseline for the training configuration |
| `train_low_lr` | Independent short continuation from epoch 1 |
| `export` | Create a new clean export from an explicitly selected checkpoint |
| `offline_smoke` | Test standard local loading in offline mode |
| `check_folder` | Run the supplied file/header checker on the exact final folder |
| `package` | Use the supplied packager after public revision verification |

`train_fixed_share` exists but its tested branch was rejected. Do not include it in routine recovery.

Recommended fresh-session sequence for reproducing training:

```text
preflight → baseline → compare → reference → train_low_lr
    → BACK UP AND DOWNLOAD → review candidate → export
    → offline_smoke → add README/license → check_folder
```

The notebook may reuse matching saved records. `reference` alone does not satisfy the training guard requiring `compare`. Never manufacture those records.

### Local evaluation configuration

```python
"EVALUATION_MODE": "exploratory",
"LOCAL_POLICY": "full",
"LOCAL_BATCH_SIZE": 1,
"LOCAL_PRECISION": "amp_fp16",
"RUNNER_CONFIRMED": False,
```

Keep the last flag false: the supplied starting kit did not reveal hidden-runner settings.

## Experiment record

### Epoch-1 comparison, reproduced on Kaggle

| Policy | Batch | Diagnostic |
|---|---:|---:|
| Centered eight seconds | 4 | 0.5635497236 |
| Centered eight seconds | 1 | 0.5556165229 |
| Full clips | 1 | 0.5684942873 |

All used AMP fp16. The results demonstrate batching sensitivity and a local benefit from retaining full audio. They do not configure the hidden runner.

### Current selected continuation

Fresh optimizer, LR `5e-6`, original square-root sampler, batch 4, random eight-second crops, maximum 800 updates. Evaluation uses the local configuration above.

| Update | Diagnostic | Change from local baseline |
|---:|---:|---:|
| 400 | 0.5744864501 | +0.005992 |
| **800** | **0.5806749508** | **+0.012181** |

This rerun replaced a lost historical branch that scored 0.5851436282. **Do not attribute that historical value to the current weights.**

### Rejected fixed-share branch

Step 400 scored 0.5347136148. The regression guard stopped it before step 800. Efik and Ibibio stayed at zero F1. This rejects the tested recipe, not all possible class balancing.

Do not extend training or tune Nupe biases based only on the aggregate diagnostic. Nupe recall is unavailable. Candidate selection must also inspect class-level regressions.

## Current candidate provenance

These paths identify the completed session; they are not portable paths or proof that files still exist:

```text
Selected checkpoint:
/kaggle/working/dsn_lid_review/training/20261021T122014418414Z/checkpoint-step-800

Clean export:
/kaggle/working/dsn_lid_review/exports/20261021T123444146370Z/model
```

Selected export hashes before documentation was added:

```text
config.json
a3856d7e98a6fbbcd79ddec4bf1aeff3359ce7a85c6b035a8ada3c33d5887522

model.safetensors
93c9ff63efc922ce83ccb831e4a3cd37710bf2dbb2aac783c8da6323e112e736

preprocessor_config.json
8cdfd65ff4115423185a1512bdae100e2e0cd744f5b322417429944aaafd0827
```

Export versions: PyTorch 2.11.0+cu128; Transformers 5.16.1; datasets 4.8.5; huggingface-hub 1.29.0; scikit-learn 1.6.1; soundfile 0.14.0; safetensors 0.8.0. These describe the tested environment, not guaranteed evaluator support.

## Export and backup rules

Set `SELECTED_MODEL_DIR` to the actual selected checkpoint, set `ALLOW_CANDIDATE_EXPORT=True`, and use fp32 or fp16. Keep `BASELINE_DIR` unchanged. An empty selection in this notebook exports the original baseline; do not leave it empty when exporting the selected continuation.

After export, set `EXPORT_FOLDER` to the newly printed path. Add documentation to that folder and rerun the checker. Do not run export again merely to add documentation: it creates another folder.

The final documented folder contains:

```text
LICENSE.md
README.md
config.json
model.safetensors
preprocessor_config.json
```

Source documentation, metadata, scripts, CSV reports and backup ZIPs must remain outside that folder. If a model-card update is made, rerun the checker; its prior folder signature may no longer match.

**Backups:** treat `/kaggle/working` as temporary. Immediately archive useful weights plus provenance and download the archive privately. Creating a ZIP without downloading/persisting it is insufficient. Notebook logs do not contain the weights. Inspect the archive and preserve hashes.

## Checks still required before competition submission

The selected export passed offline CPU functional loading and file/header checks at approximately 94.6M parameters, 0.378 GB and F32. Those passes do not establish speed or memory compliance.

- Verify the final public HF repository at its pinned revision using standard classes.
- Confirm the exact ID-to-label order, not just the set of label names.
- Resolve evaluator-version compatibility where possible.
- Check whole-pipeline timing, peak RAM/GPU memory and total runtime under the official procedure; local AMP timings are preliminary only.
- Keep mixed-source attribution and the supplied license notice. Do not invent a blanket model license.
- Ask organisers about any unresolved licensing or evaluator behavior.
- Leave the submitted public repository unchanged after submission.

The competition ZIP contains only `submission.json` at its root:

```json
{"repo": "USERNAME/MODEL", "revision": "REAL_40_CHARACTER_COMMIT_HASH"}
```

Use the supplied `make_submission.py` and its documented arguments. A model backup ZIP is not a submission ZIP. The supplied `check_model_folder.py` does not measure speed or memory, and its label-membership check does not replace exact index-order assertions.

Official F1 below 0.60 may be displayed but is not ranked. The threshold applies to official all-class F1, not the final efficiency-adjusted score or this Nupe-missing local diagnostic.

## Troubleshooting

| Symptom | Action |
|---|---|
| Cannot read HF secret | Enable the existing secret for the current notebook; never print it |
| Dataset access error | Check Internet, account permission and token scope; retain the pinned revision |
| Missing baseline folder | Check the private input attachment and path; do not create an empty replacement |
| Missing comparison guard | Run `compare` in this workspace; do not bypass the guard |
| Missing export/candidate | Inspect saved/private backups; do not silently substitute epoch 1 |
| ZIP already exists | Verify its contents/hashes or create a new timestamped archive |
| Old export selected after Run All | Update the main configuration, not only a temporary helper cell |
| Earlier helper actions rerun | Disable standalone action cells; the dispatcher cannot control them |

## References

- Competition: https://www.codabench.org/competitions/18424/
- Official data: https://huggingface.co/datasets/Data-Science-Nigeria/dsn-2026-lid-train
- Pretrained checkpoint: https://huggingface.co/facebook/wav2vec2-base
