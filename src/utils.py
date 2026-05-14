import pandas as pd

from src.config import TIER_BOUNDARIES


def make_tier_labels(y_raw):
    """Bin raw SalePrice values into 4 tier labels (0=Low, 1=Medium, 2=High, 3=Luxury)."""
    bins = [0] + TIER_BOUNDARIES + [float("inf")]
    return pd.cut(y_raw, bins=bins, labels=[0, 1, 2, 3]).astype(int)


def print_section(title):
    """Print a formatted section header to the console."""
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")
