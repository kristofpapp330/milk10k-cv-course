# Model card — MILK10k 11-class skin-lesion classifier (EADA-CV26-KP)

Course project, Computer Vision and Speech Recognition (EADA), Kristof Papp, November 2026.
**Research and teaching only. Not a medical device. Not for diagnosing patients.**

## Model
| | |
|---|---|
| Architecture | ResNet-18, ImageNet-1k weights from torchvision (`ResNet18_Weights.IMAGENET1K_V1`) |
| What was trained on MILK10k | `layer3`, `layer4` and a new head `nn.Linear(512, 11)`; conv1–layer2 stay as pretrained |
| File | `m5_final/model_11class.pt` (42 MB, only the trained weights; the rest is reloaded from torchvision) |
| Training | class-weighted cross-entropy, Adam (1e-3 head, 1e-4 pretrained layers), batch 64, augmentation = flips, ±20° rotation, mild brightness/hue jitter; early stopping on validation macro-F1 (best epoch 5); Google Colab T4 GPU |
| Decision thresholds | one per class, tuned on the validation split; stored in the checkpoint and applied by `make_submission.py` as a rescaling (threshold → 0.5) |

## Input and output
- **Input:** one **dermoscopic** JPEG of a single lesion. Clinical (non-dermoscopic) photos were never used for training.
- **Preprocessing (identical for validation, test, `predict()` and the submission):** RGB → resize to 224 × 224 →
  ImageNet mean/std. Code: `milk10k_pipeline/deep.py` (`prepare_image`, `VAL_TRANSFORM`).
- **Output:** 11 softmax probabilities, in the order AKIEC, BCC, BEN_OTH, BKL, DF, INF, MAL_OTH, MEL, NV, SCCKA, VASC.
- **Use:** `python m5_final/predict.py image.jpg`, or `from m5_final.predict import predict`.

## Data
- MILK10k (ISIC Archive): 5,240 lesions, one dermoscopic image per lesion used. Licence CC-BY-NC.
  MILK study team, MILK10k Benchmark, ISIC Archive 2025, doi:10.34970/262082; Tschandl P et al., J Invest Dermatol 2026;146(2):357-364.
- **Split by lesion** (no lesion in two splits), stratified on the 11 diagnoses, seed 42: 3,143 train / 1,048 validation / 1,049 test.
- **Strong imbalance:** BCC is about 48 % of the lesions; MAL_OTH has 5 training lesions, BEN_OTH 26, VASC 28, DF 30, INF 30.
- **Skin tone (0 = very dark … 5 = very light):** 60 % of the test lesions are tone 3; only 24 test lesions (2.3 %) are tone 0–1.

## Performance
| | macro-F1 |
|---|---|
| 11 classes, validation, argmax | 0.356 |
| 11 classes, our test split, official 0.5 rule with tuned thresholds (local estimate) | 0.389 |
| **11 classes, official MILK10k leaderboard (479 blind lesions)** | **0.362** |
| Same architecture with 3 outputs (`diagnosis_1`), test split | 0.578 (95 % CI 0.523–0.634) |

- **Per class (validation F1):** NV 0.68, BCC 0.66, VASC 0.57, SCCKA 0.55, MEL 0.48, AKIEC 0.29, BKL 0.29,
  DF 0.17, INF 0.12, BEN_OTH 0.10, MAL_OTH 0.00.
- **3-class sibling as a triage tool** ("needs clinical action" = Malignant or Indeterminate): sensitivity 0.73 and
  specificity 0.73 on test. It **catches only 38 % of melanomas**.

## Intended use
- Teaching and research on decision support for dermoscopy.
- A benchmark entry for the MILK10k challenge.
- A *second opinion* that could flag lesions for a clinician's attention in a research setting.

## Must not be used
- To diagnose a patient, to rule out skin cancer, or to decide whether to biopsy or treat a lesion.
- On clinical close-up or smartphone photos, on images from other devices or populations, or on skin tones that are
  barely represented in MILK10k (tones 0–1).
- As a calibrated risk score: the probabilities are overconfident.

## Known limitations and risks
- **Misses many cancers:** over a quarter of malignant lesions and most melanomas (3-class model on test).
- **Overconfident:** the most confident mistakes have probability ≈ 1.00, and the validation loss rises during
  training. The model is not calibrated.
- **Shortcut risk:** errors cluster on images with surgical-ink marks, the dark dermoscope frame, gel bubbles and
  hairs.
- **Rare classes are close to unlearned:** MAL_OTH, BEN_OTH and INF have F1 ≈ 0–0.12. Their tuned thresholds rest
  on 2–11 validation lesions.
- **Fairness:** no significant difference between skin-tone groups or sexes on our test split (all confidence
  intervals overlap). Performance on very dark skin is **unknown**: there are only 24 test lesions of tone 0–1.
  The sex gap (macro-F1 0.65 women vs 0.54 men) should be monitored.
- **One training run (seed)** per model. The differences between the top models are within that noise.
- **Generalisation:** official score 0.362 vs local estimate 0.389. New data from other clinics or devices may
  be worse still.

## Reproduce
See `README.md`. Notebooks `m3_deep_models` → `m4_best_model` → `m5_final_11class` (Google Colab, T4 GPU,
about 45 minutes in total), then `python m5_final/make_submission.py`. Tests: `python tests/test_deep.py`.
