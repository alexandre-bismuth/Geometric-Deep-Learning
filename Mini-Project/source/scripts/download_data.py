#!/usr/bin/env python3
"""Pre-download all datasets so training doesn't block on downloads.

Usage:
    python scripts/download_data.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_geometric.datasets import ZINC, LRGBDataset

DATA_ROOT = "data"


def main():
    print("Pre-downloading all datasets...\n")

    # 1. ZINC-12k
    print("=== ZINC-12k ===")
    for split in ['train', 'val', 'test']:
        ds = ZINC(root=os.path.join(DATA_ROOT, 'ZINC'), subset=True, split=split)
        print(f"  {split}: {len(ds)} graphs")

    # 2. Peptides-func
    print("\n=== Peptides-func ===")
    for split in ['train', 'val', 'test']:
        ds = LRGBDataset(root=os.path.join(DATA_ROOT, 'LRGB'), name='Peptides-func', split=split)
        print(f"  {split}: {len(ds)} graphs")

    # 3. PascalVOC-SP
    print("\n=== PascalVOC-SP ===")
    for split in ['train', 'val', 'test']:
        ds = LRGBDataset(root=os.path.join(DATA_ROOT, 'LRGB'), name='PascalVOC-SP', split=split)
        print(f"  {split}: {len(ds)} graphs")

    print("\nAll datasets downloaded successfully!")
    print(f"Data root: {os.path.abspath(DATA_ROOT)}")


if __name__ == "__main__":
    main()
