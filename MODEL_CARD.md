---
pipeline_tag: audio-classification
base_model: facebook/wav2vec2-base
datasets:
- Data-Science-Nigeria/dsn-2026-lid-train
tags:
- language-identification
- wav2vec2
- nigerian-languages
---

# DSN 2026 Nigerian Language Identification — Low-LR Candidate

A standard Hugging Face audio-classification model fine-tuned from `facebook/wav2vec2-base` on the official DSN competition training split. This model card documents the selected recovery-run checkpoint, not the original epoch-1 model or the earlier lost continuation.

## Model summary

| Item | Value |
|---|---|
| Architecture | Wav2Vec2 audio classification |
| Parameters | 94,571,917 |
| Export | fp32 safetensors |
| Input | 16 kHz mono audio; competition clips are 2–30 seconds |
| Selected checkpoint | Recovery-run low-LR continuation, update 800 |
| Local diagnostic | **0.5806749508**, with Nupe unmeasured |

### Exact label mapping

| Index | Code | Language |
|---:|---|---|
| 0 | yor | Yoruba |
| 1 | ibo | Igbo |
| 2 | hau | Hausa |
| 3 | ego | Eggon |
| 4 | anw | Annang |
| 5 | efi | Efik |
| 6 | ibb | Ibibio |
| 7 | idu | Idoma |
| 8 | igl | Igala |
| 9 | nup | Nupe |
| 10 | tiv | Tiv |
| 11 | urh | Urhobo |
| 12 | other | Speech outside the 12 target languages |

## Data and training

- Official dataset: [Data-Science-Nigeria/dsn-2026-lid-train](https://huggingface.co/datasets/Data-Science-Nigeria/dsn-2026-lid-train).
- Pinned dataset revision: `87cf6d4b1c7599f97f44b6e6aade86438369a2b1`.
- 24,774 training clips and 4,352 validation clips.
- No shared clip IDs or known `(source, speaker_id)` pairs were found between splits. This does not rule out duplicate audio under different IDs or overlap where metadata is missing.
- Only the training split was used for gradient updates. Validation was used for evaluation and checkpoint selection.
- No additional external training dataset was introduced. The pretrained initialization is [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base).
- The original upstream corpora named in the dataset attribution were not separately downloaded or used directly as additional training data.

Original fine-tuning used batch size 4, AdamW at `3e-5`, per-example sampling weights proportional to `1/sqrt(class_count)`, and random crops up to eight seconds. Epoch 1 was retained because later epochs worsened the original validation diagnostic.

The selected model continues from epoch 1 for 800 updates with a fresh optimizer at `5e-6`, the original sampler, batch size 4, and random crops up to eight seconds. The convolutional feature encoder was frozen during continuation. Training used AMP fp16; exported parameters are fp32.

## Preprocessing and loading

The feature extractor uses 16 kHz audio, normalization, right-side zero padding, and `return_attention_mask=False`. Preserve the saved preprocessing configuration rather than enabling masks indiscriminately.

```python
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

# Set these to the actual published repository and verified commit.
repo_id = "YOUR_USERNAME/YOUR_MODEL"
revision = "YOUR_FULL_40_CHARACTER_COMMIT_HASH"

extractor = AutoFeatureExtractor.from_pretrained(
    repo_id, revision=revision, trust_remote_code=False
)
model = AutoModelForAudioClassification.from_pretrained(
    repo_id, revision=revision, trust_remote_code=False
).eval()
```

The snippet contains placeholders; it does not identify an already verified public repository. For offline use, download the required files beforehand and load their local folder with `local_files_only=True`.

No custom cropping, window aggregation, or logit calibration is embedded in this repository. Local full-clip evaluation settings do not automatically configure the organiser's runner.

## Local evaluation — not an official competition score

Evaluation used full clips, batch size 1, and AMP fp16.

| Checkpoint | Represented-class diagnostic |
|---|---:|
| Original epoch 1 | 0.5684942873 |
| Recovery continuation, update 400 | 0.5744864501 |
| **Selected recovery continuation, update 800** | **0.5806749508** |

The diagnostic is the fixed-share mean of per-class F1 normalized over classes with validation support. The stated shares are 0.043 for Yoruba, Igbo and Hausa and 0.087 for each remaining class. The rounded shares sum to 0.999; represented validation share mass is 0.912.

**Nupe has zero validation clips. Its recall and F1 are unmeasured. This diagnostic must not be compared directly with the official 0.60 ranking threshold.** Predictions into Nupe still count as errors for true represented classes. No Nupe-specific calibration was applied.

Selected checkpoint F1 values:

| yor | ibo | hau | ego | anw | efi | ibb | idu | igl | nup | tiv | urh | other |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| .837370 | .918919 | .851852 | .862423 | .896391 | .000000 | .000000 | .831785 | .098361 | Unmeasured | .310734 | .925170 | .873129 |

## Checks and limitations

The selected export passed standard Hugging Face offline CPU functional loading and the supplied starting-kit file/header checker. The checker reported approximately 94.6M parameters, 0.378 GB and F32 weights. Exact label-order assertions also passed in the notebook.

These checks do **not** certify inference speed, peak RAM/GPU memory, total runtime, hidden-runner compatibility, or ranking eligibility. The supplied kit did not contain the hidden inference/scoring runner. Local export versions included Transformers `5.16.1` and PyTorch `2.11.0+cu128`; compatibility with the evaluator's environment remains to be verified.

Efik and Ibibio remain failures in local validation; Igala and Tiv remain weak. Several classes have very few speakers, and Nupe has one known training speaker. Gains in some classes accompany losses in others, including Hausa and Annang. No winning result or suitability for high-stakes speaker decisions is claimed.

## Attribution and licensing

The official dataset declares mixed per-source licensing. Preserve its supplied `LICENSE.md` unchanged with the model. The supplied `ATTRIBUTION.md` states:

- **afrispeech_ipa**: CC-BY-4.0 (stated; upstream audio terms not verified).
- **hypa_voices**: Apache-2.0.
- **omnilingual_asr**: CC-BY-4.0.

Pinned source documents:
- [ATTRIBUTION.md](https://huggingface.co/datasets/Data-Science-Nigeria/dsn-2026-lid-train/blob/87cf6d4b1c7599f97f44b6e6aade86438369a2b1/ATTRIBUTION.md)
- [LICENSE.md](https://huggingface.co/datasets/Data-Science-Nigeria/dsn-2026-lid-train/blob/87cf6d4b1c7599f97f44b6e6aade86438369a2b1/LICENSE.md)
- [Dataset card](https://huggingface.co/datasets/Data-Science-Nigeria/dsn-2026-lid-train/blob/87cf6d4b1c7599f97f44b6e6aade86438369a2b1/README.md)

The dataset card documents resampling, segmentation, label mapping, deduplication and speaker-based splitting. This model adds supervised fine-tuning and continued fine-tuning; source audio is not redistributed here.

The pretrained model card declares Apache-2.0. This does not establish a blanket license for all data rights or derivatives. No single global license is asserted here. The supplied qualification about unverified upstream terms remains unresolved; seek organiser clarification if necessary.

The dataset documentation requests this research citation:

```bibtex
@misc{omnilingualasr2025,
    title={{Omnilingual ASR}: Open-Source Multilingual Speech Recognition for 1600+ Languages},
    author={{Omnilingual ASR Team} and others},
    year={2025},
    eprint={2511.09690},
    archivePrefix={arXiv},
    primaryClass={cs.CL},
    url={https://arxiv.org/abs/2511.09690}
}
```
