# Submissions — Homework Sessions 1–3, Project Milestone 1 and Milestone 2

**Name:** Kristof Papp
**Repository:** https://github.com/kristofpapp330/milk10k-cv-course

## Part A
- Notebook: [notebooks/homework_part_a.ipynb](notebooks/homework_part_a.ipynb)
- PDF: [notebooks/homework_part_a.pdf](notebooks/homework_part_a.pdf)

## Part B — Milestone 1
| item | file |
|---|---|
| B0 README | [README.md](README.md) |
| requirements | [requirements.txt](requirements.txt) |
| source code (package) | [milk10k_pipeline/](milk10k_pipeline/) |
| pipeline script | [run_milestone1.py](run_milestone1.py) |
| tests | [tests/test_pipeline.py](tests/test_pipeline.py) |
| B1 integrity check code | [milk10k_pipeline/integrity.py](milk10k_pipeline/integrity.py) |
| B1 image size table | [outputs/image_size_summary.csv](outputs/image_size_summary.csv) |
| B2 class distribution figure | [outputs/figures/class_distribution.png](outputs/figures/class_distribution.png) |
| B2 11-class → diagnosis_1 mapping | [outputs/figures/dx_vs_diagnosis1.png](outputs/figures/dx_vs_diagnosis1.png) |
| B2 Session 2 findings table | [outputs/session2_findings.csv](outputs/session2_findings.csv) |
| B2 class gallery | [outputs/figures/class_gallery.png](outputs/figures/class_gallery.png) |
| B3 label strategy code | [milk10k_pipeline/labels.py](milk10k_pipeline/labels.py) |
| B3 label map | [outputs/label_map.json](outputs/label_map.json) |
| B4 data-quality report | [outputs/data_quality_report.md](outputs/data_quality_report.md) |
| B5 split code | [milk10k_pipeline/splits.py](milk10k_pipeline/splits.py) |
| B5 train split | [outputs/splits/train.csv](outputs/splits/train.csv) |
| B5 validation split | [outputs/splits/val.csv](outputs/splits/val.csv) |
| B5 test split | [outputs/splits/test.csv](outputs/splits/test.csv) |
| B5 split info (seed, date) | [outputs/splits/split_info.json](outputs/splits/split_info.json) |
| B5 class proportions per split | [outputs/split_class_proportions.csv](outputs/split_class_proportions.csv) |
| B7 transforms | [milk10k_pipeline/transforms.py](milk10k_pipeline/transforms.py) |
| B7 augmentation figure | [outputs/figures/augmentation_examples.png](outputs/figures/augmentation_examples.png) |
| normalization statistics | [outputs/normalization_stats.json](outputs/normalization_stats.json) |
| B8 Dataset / DataLoaders | [milk10k_pipeline/data_loader.py](milk10k_pipeline/data_loader.py) |
| B8 class weights | [outputs/class_weights.json](outputs/class_weights.json) |
| B8 batch of 16 after transforms | [outputs/figures/train_batch.png](outputs/figures/train_batch.png) |
| B8 printed sanity checks (full run log) | [outputs/run_log.txt](outputs/run_log.txt) |
| B9 report | [reports/milestone1_report.md](reports/milestone1_report.md) |

## Milestone 2 — Classical ML baseline (Session 5)
| item | file |
|---|---|
| notebook: experiments + report | [notebooks/classical_baseline.ipynb](notebooks/classical_baseline.ipynb) |
| notebook exported to PDF | [notebooks/classical_baseline.pdf](notebooks/classical_baseline.pdf) |
| feature extraction code | [milk10k_pipeline/features.py](milk10k_pipeline/features.py) |
| models, selection, metrics code | [milk10k_pipeline/classical.py](milk10k_pipeline/classical.py) |
| tests | [tests/test_features.py](tests/test_features.py) |
| feature table (all 5,240 dermoscopic images) | [outputs/baseline/features_dermoscopic.csv](outputs/baseline/features_dermoscopic.csv) |
| validation results, every configuration (binary) | [outputs/baseline/val_results_binary.csv](outputs/baseline/val_results_binary.csv) |
| validation results, every configuration (3-class) | [outputs/baseline/val_results_3class.csv](outputs/baseline/val_results_3class.csv) |
| final test results | [outputs/baseline/test_results.json](outputs/baseline/test_results.json) |
| figures | `outputs/figures/baseline_*.png` |

## Session 6 homework — MLP
| item | file |
|---|---|
| notebook (scoreboard, nn.Conv2d, MLP on raw pixels) | [notebooks/homework_mlp.ipynb](notebooks/homework_mlp.ipynb) |
| notebook exported to PDF | [notebooks/homework_mlp.pdf](notebooks/homework_mlp.pdf) |
| MLP + training loop code | [milk10k_pipeline/mlp.py](milk10k_pipeline/mlp.py) |
| tests | [tests/test_mlp.py](tests/test_mlp.py) |
| scoreboard | [outputs/mlp/scoreboard.csv](outputs/mlp/scoreboard.csv) |
| every MLP run (validation) | [outputs/mlp/val_results_all_runs.csv](outputs/mlp/val_results_all_runs.csv) |

