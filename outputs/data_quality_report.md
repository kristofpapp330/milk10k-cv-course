# Data-quality report (B4)

Generated from `metadata.csv`: 10480 images, 5240 lesions.

## 1. Missing values and how each column is handled

| column | % missing | decision |
|---|---|---|
| anatom_site_special | 98.0 | leave NaN; not used as a model input |
| diagnosis_4 | 85.5 | leave NaN; label column, never an input |
| melanocytic | 77.2 | leave NaN; not used as a model input |
| anatom_site_general | 37.3 | explicit 'unknown' category; missingness is informative (NV twice as frequent, Part A Q4), so it is NOT used as input in Milestone 1 |
| diagnosis_3 | 1.5 | leave NaN; label column, never an input |
| age_approx | 0.4 | impute with the TRAIN median (only if age is used as an input later) |

## 2. Label-consistency checks

- lesions: **5240**
- lesions with exactly 2 images: **5240**
- lesions with 1 dermoscopic + 1 clinical: **5240**
- the two images of each lesion have identical age, sex, site and diagnosis fields (checked in Part A, A1.1e)

## 3. Shortcut check

### image_manipulation vs diagnosis_1

Counts:

| image_manipulation | Benign | Indeterminate | Malignant | All |
|---|---|---|---|---|
| altered | 133 | 43 | 159 | 335 |
| instrument only | 2833 | 203 | 7109 | 10145 |
| All | 2966 | 246 | 7268 | 10480 |

Row percentages:

| image_manipulation | Benign | Indeterminate | Malignant |
|---|---|---|---|
| altered | 39.7 | 12.8 | 47.5 |
| instrument only | 27.9 | 2.0 | 70.1 |

### image_type vs diagnosis_1

Counts:

| image_type | Benign | Indeterminate | Malignant | All |
|---|---|---|---|---|
| clinical: close-up | 1483 | 123 | 3634 | 5240 |
| dermoscopic | 1483 | 123 | 3634 | 5240 |
| All | 2966 | 246 | 7268 | 10480 |

Row percentages:

| image_type | Benign | Indeterminate | Malignant |
|---|---|---|---|
| clinical: close-up | 28.3 | 2.3 | 69.4 |
| dermoscopic | 28.3 | 2.3 | 69.4 |

**Conclusion.**
- `image_manipulation`: 47.5% of *altered* images are Malignant vs 70.1% of the others. The editing status of an image is not biology, so any difference here is a potential shortcut: the column is banned and all images get the same preprocessing.
- `image_type`: the class percentages are identical for dermoscopic and clinical images (69.4% vs 69.4% Malignant), because every lesion has exactly one image of each type. Image type carries no label information by construction, so it is not a shortcut.

## 4. Columns NOT used as model inputs

| column | reason |
|---|---|
| diagnosis_1 | the target itself |
| diagnosis_2 | finer version of the label |
| diagnosis_3 | finer version of the label |
| diagnosis_4 | finer version of the label |
| diagnosis_5 | finer version of the label |
| dx | the 11-class label |
| diagnosis_confirm_type | result of the diagnostic process (histopathology = suspicious lesion, 72% malignant) |
| image_manipulation | acquisition/editing artefact that differs between classes -> shortcut |
| attribution | identifies the contributing clinic -> site/acquisition shortcut |
| copyright_license | tied to the contributor -> acquisition shortcut |
| lesion_id | identifier (needed only for grouping the split) |
| isic_id | identifier |
