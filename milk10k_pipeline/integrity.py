"""
B1 -- Full-dataset integrity check.

Every metadata row must resolve to an existing, readable image file.
"""
import pandas as pd
from PIL import Image


def check_images(meta, image_paths):
    """Check every row of the per-image metadata.

    For each isic_id: does the file exist, does Image.verify() pass, and what is its width/height.
    Returns a per-image DataFrame with columns
    isic_id, view, exists, readable, error, width, height.
    """
    rows = []
    for isic_id, view in zip(meta["isic_id"], meta["view"]):
        path = image_paths.get(isic_id)
        row = {"isic_id": isic_id, "view": view, "exists": path is not None,
               "readable": False, "error": "", "width": None, "height": None}
        if path is not None:
            try:
                with Image.open(path) as img:
                    row["width"], row["height"] = img.size   # read from the header
                    img.verify()                             # checks the file structure
                row["readable"] = True
            except Exception as error:                       # any decoding problem counts as unreadable
                row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return pd.DataFrame(rows)


def size_summary(checks):
    """Min / median / max width and height, per image type and overall (the image_size_summary.csv table)."""
    groups = [("all", checks)] + list(checks.groupby("view"))
    rows = []
    for name, g in groups:
        ok = g[g["readable"]]
        rows.append({"image_type": name,
                     "n_rows": len(g),
                     "n_missing": int((~g["exists"]).sum()),
                     "n_unreadable": int((g["exists"] & ~g["readable"]).sum()),
                     "width_min": ok["width"].min(), "width_median": ok["width"].median(), "width_max": ok["width"].max(),
                     "height_min": ok["height"].min(), "height_median": ok["height"].median(), "height_max": ok["height"].max()})
    return pd.DataFrame(rows)
