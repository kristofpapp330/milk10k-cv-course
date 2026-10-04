# Milestone 1 report — MILK10k data pipeline

Kristof Papp · splits created 2026-10-04 with seed 42 · all numbers come from `outputs/run_log.txt`

## 1. Data and integrity (B1)
MILK10k has 5,240 lesions × 2 images (dermoscopic + clinical close-up) = 10,480 JPEGs. All 10,480 metadata rows resolve to
an existing file and all pass `Image.verify()` (0 missing, 0 unreadable). Every image is exactly 600×450 px
(`outputs/image_size_summary.csv`). The loaders now **fail loudly**: a missing file raises `FileNotFoundError` unless
`allow_missing=True` is passed explicitly.

## 2. Labels and biopsy enrichment (B2, B3)
| diagnosis_1 | lesions | | 11 classes (lesions) |
|---|---|---|---|
| Malignant | 3,634 | | BCC 2,522 · NV 746 · BKL 544 · SCCKA 473 · MEL 450 · AKIEC 303 |
| Benign | 1,483 | | DF 52 · INF 50 · VASC 47 · BEN_OTH 44 · MAL_OTH 9 |
| Indeterminate | 123 | | |

Ten classes map to exactly one `diagnosis_1` value; **AKIEC** is split into 180 Malignant and 123 Indeterminate lesions,
which are all of the Indeterminate lesions.

**Biopsy enrichment.** MILK10k is not a sample of the lesions a doctor sees. 96% of the lesions (5,016) were confirmed by
histopathology, which means someone found them suspicious enough to biopsy, and 72% of these are malignant. The 224 lesions
confirmed only by clinical assessment are 2% malignant. As a result, 69% of the lesions in the dataset are malignant, while
in a dermatology or primary-care clinic most examined lesions are benign. Two consequences for the final model:
(1) **accuracy is not transferable**: a model that always says "Malignant" already reaches ~70% accuracy here, and in a
clinic the same model would be wrong most of the time. Precision (PPV) also depends directly on prevalence, so a PPV measured
on MILK10k will be far too optimistic in practice. (2) The benign lesions in the data are the *hard* benign ones (those that
looked suspicious), so the model learns a harder boundary than the easy everyday cases. For these reasons I will report
**sensitivity for malignant, specificity and balanced accuracy / ROC-AUC**, which do not depend on prevalence. Any
deployment claim would need re-calibration on data with real-world prevalence.

**Session 2 findings on the full dataset** (`outputs/session2_findings.csv`):

| Session 2 finding | full dataset | still true? | consequence |
|---|---|---|---|
| strong class imbalance | BCC 2,522 vs MAL_OTH 9 (280:1) | yes | grouped stratified split, class weights + sampler, balanced metrics |
| malignant patients are older | mean age 65.1 vs 51.6 years | yes | real risk factor, possible later input (train-median imputation) |
| site is associated with diagnosis | missing site: NV 21.1% vs 10.1% | yes, missingness informative | potential shortcut → site not an input in M1 |
| mean colour does not separate classes | brightness ROC-AUC 0.457 | yes | needs a CNN; colour augmentation kept mild |
| Indeterminate images darker | 141.5 vs 150–153 | yes (small group) | no special handling; class merged |

**Label strategy (B3).** *Primary:* Indeterminate is **merged into Malignant**. There are too few for a reliable third
class (123 lesions, 2.3%), they are all AKIEC and so cannot be told apart visually from the Malignant AKIEC, and clinically
an indeterminate lesion needs the same action (work-up/biopsy). Merging errs on the safe side. Dropping them would remove
cases the model will meet in practice. The result is Benign 1,483 vs Malignant 3,757. *Stretch (11 classes):* **all
classes are kept** and the rare ones (DF, INF, VASC, BEN_OTH, MAL_OTH) are handled with class weights. Merging them into
one "other" class would make training easier but would mix benign (DF, INF, VASC, BEN_OTH) and malignant (MAL_OTH)
lesions, which is clinically meaningless. The cost of keeping them is very noisy metrics: MAL_OTH has 1–2 lesions in val
and test, so its results are reported with a warning. The mapping is in `outputs/label_map.json`.

## 3. Data quality (B4)
Every lesion has exactly 2 images, 1 of each type (5,240/5,240). Missing values: `anatom_site_general` 37.3% (explicit
"unknown", not used as an input), `age_approx` 0.4% (train-median imputation if used), sparse columns left as NaN.
**Shortcut check:** `image_manipulation` = *altered* (335 images) is 47.5% Malignant and 12.8% Indeterminate, against
70.1% and 2.0% for unaltered images. Image editing is linked to the label, so the column is **banned**. `image_type` has
identical class rates by construction (one image of each type per lesion), so it is not a shortcut. **Banned inputs:**
`diagnosis_1–4`, `dx`, `melanocytic` (part of the diagnosis), `diagnosis_confirm_type` (a result of the diagnostic
process), `image_manipulation`, `attribution`/`copyright_license` (identify the contributor), and the IDs.
Full report: `outputs/data_quality_report.md`.

## 4. Splits (B5)
`StratifiedGroupKFold` on the **lesion table**, grouped by `lesion_id` and stratified on the 11-class label, 60/20/20.
Images are assigned to their lesion's split afterwards. Train 3,143 / val 1,048 / test 1,049 lesions. Checks printed in
code: 0 lesion overlap for every pair of splits, every lesion has its 2 images in its own split, the maximum deviation of
any class proportion from the global one is **0.084 percentage points**, and `diagnosis_1` proportions are within 0.5 pp
across splits. Part A showed why grouping matters: with an image-level split 80% of test images have their sibling in
train, and a "sibling oracle" reaches 0.85 balanced accuracy.

## 5. Preprocessing, augmentation, loading (B6–B8)
- **Resolution 224×224.** All images are 600×450, so 100% are larger than the input: resizing the shorter side to 256
  and centre-cropping 224 only down-samples, and matches ImageNet-pretrained CNNs. The crop removes some of the image's
  left and right borders, which is acceptable because dermoscopic lesions are centred.
- **Views:** option **(b) dermoscopic only**. Dermoscopy is the standardised modality, so one image corresponds to one
  lesion and nothing is double-counted at evaluation. (a) *Both images as independent samples* doubles the data, but
  predictions must then be averaged per lesion, and a per-image evaluation would count each lesion twice. With a
  non-grouped split it would also leak, since the model would see one view in train and the other in test.
  (c) *Both views per lesion* (two-branch fusion) is the most informative and is prepared by the `LesionDataset` of
  Part A (A3.6). It is a candidate for a later milestone.
- **Normalization:** mean (0.674, 0.535, 0.529), std (0.132, 0.136, 0.150), computed on the **train split only**
  (`outputs/normalization_stats.json`).
- **Augmentation (train only):** random 224 crop from 256, horizontal and vertical flips, ±20° rotation, ColorJitter
  with brightness 0.1 and hue 0.02. Skin has no up/down or left/right. The colour limits come from Part A (A3.5): the
  real Benign–Malignant gap is only 6.6° of hue and 0.074 of brightness, and hue ≥ 0.05 or brightness ≥ 0.3 shifts
  colours more than that. The eval transform is deterministic, which is asserted in code. Justification table:
  `milk10k_pipeline/transforms.py`; figure: `outputs/figures/augmentation_examples.png`.
- **Imbalance (train):** 2,254 Malignant vs 889 Benign = **2.54:1** (11 classes: BCC 1,514 vs MAL_OTH 5 = 303:1).
  Two mechanisms are implemented: class weights for the loss (Benign 1.77, Malignant 0.70, `outputs/class_weights.json`)
  and a `WeightedRandomSampler`. Over 20 sampled train batches the label histogram is 330 Benign / 310 Malignant
  (≈50/50, against 28/72 without the sampler). Later milestones will use **one** of the two, so the correction is not
  applied twice.
- **DataLoaders:** batch 32, `num_workers=0` (portable on Windows and in notebooks), seeds set for Python, NumPy and
  PyTorch. Batch shape (32, 3, 224, 224), float32, values in about [−5.1, 3.1] after normalization. One training epoch
  loads in ≈49 s. Figure: `outputs/figures/train_batch.png`.

## 6. What could still go wrong
**Residual leakage:** splits are grouped by lesion, not by **patient**. If two lesions come from the same person (same
skin, same camera, same clinic), they can end up in train and test and inflate the scores. Grouping by patient would need a
patient identifier in the metadata. Near-duplicate
photos across lesions were not searched for. **Biases:** the data is biopsy-enriched, mostly from older patients, and
comes from specific contributors and devices. Skin tone was not analysed in this milestone, and performance on darker
skin or on smartphone photos is unknown. Missing anatomical site and image editing are linked to the label, which shows
that collection artefacts exist that a CNN could exploit from the pixels as well. **What a clinician should know:** the
model is decision support for dermoscopic images only. Its accuracy is measured on suspicious, biopsied lesions and does
not represent everyday prevalence. Indeterminate lesions are reported as "needs action", and rare diagnoses (e.g.
MAL_OTH, 9 lesions) cannot be evaluated reliably.
