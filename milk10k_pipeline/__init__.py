"""
milk10k_pipeline
=================

Reusable data-analysis and preprocessing pipeline for the MILK10k
course project (Computer Vision & Speech Recognition, Session 1-2 homework
extended for the Session 2-3 homework).

Modules
-------
metadata_analysis   -- Part 1: metadata / target correlation analysis
color_analysis       -- Part 2: color & histogram analysis across the dataset
preprocessing        -- Part 3: single-image and batch preprocessing functions
data_loader          -- Part 4: MILK10kDataLoader
visualizer           -- Part 5: reusable plotting utilities (+ class gallery, tensor batches)

Added in Milestone 1 (Part B)
-----------------------------
config               -- the single place for paths, seed, image size, batch size
data_io              -- loading the CSV tables, locating image files
integrity            -- B1: existence + Image.verify() of every image, size table
labels               -- B3: label strategy / label_map.json, lesion-level table
quality              -- B4: data-quality report
splits               -- B5: leak-free lesion-level train/val/test split
transforms           -- B7: train_transform / eval_transform
data_loader          -- B8: MILK10kDataset + DataLoaders (upgrade of the Session 2 loader)
"""

from . import metadata_analysis
from . import color_analysis
from . import preprocessing
from . import data_loader
from . import visualizer
from . import config, data_io, integrity, labels, quality, splits, transforms

__all__ = [
    "metadata_analysis",
    "color_analysis",
    "preprocessing",
    "data_loader",
    "visualizer",
    "config",
    "data_io",
    "integrity",
    "labels",
    "quality",
    "splits",
    "transforms",
]
