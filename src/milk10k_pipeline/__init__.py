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
visualizer           -- Part 5: reusable plotting utilities
"""

from . import metadata_analysis
from . import color_analysis
from . import preprocessing
from . import data_loader
from . import visualizer

__all__ = [
    "metadata_analysis",
    "color_analysis",
    "preprocessing",
    "data_loader",
    "visualizer",
]
