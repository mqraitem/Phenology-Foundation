"""
Random baseline for phenology prediction.

Computes per-stage (G, M, S, D) DOY distribution from training GT at paper-mask
pixels, then generates predictions by sampling from N(mean, std) per stage.
Outputs a CSV identical in format to the model eval CSVs so it can be consumed
by the results notebook unchanged.

Usage:
    python eval_random_baseline.py --split test --data_percentage 1.0
    python eval_random_baseline.py --split val  --data_percentage 0.4
"""

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import path_config
from lib.utils import get_data_paths, get_masks_paper
from lib.dataloaders.dataloaders import load_raster_output


STAGE_NAMES = ["G", "M", "S", "D"]
# Indices into the LSP raster that correspond to the 4 phenology dates
CORRECT_INDICES = [1, 4, 7, 10]  # zero-based from [2,5,8,11]-1


def collect_training_stats(data_percentage):
    """Scan training GT at paper-mask pixels and return per-stage (mean, std) in DOY."""
    path_train = get_data_paths("training", data_percentage)
    paper_masks = get_masks_paper("train")  # returns dict {tile_name: bool mask 330x330}

    stage_values = {i: [] for i in range(4)}

    for image_paths, gt_path, tile_name in tqdm(path_train, desc="Collecting training GT stats"):
        if gt_path is None:
            continue
        if tile_name not in paper_masks:
            continue

        gt = load_raster_output(gt_path)
        gt = gt[CORRECT_INDICES, :, :]  # (4, H, W)
        mask = paper_masks[tile_name].cpu().numpy()  # (330, 330) bool

        for idx in range(4):
            stage_vals = gt[idx][mask].astype(np.float64)
            valid = (stage_vals != 32767) & (stage_vals >= 0)
            stage_values[idx].extend(stage_vals[valid].tolist())

    stats = {}
    for idx in range(4):
        arr = np.array(stage_values[idx])
        stats[idx] = (float(arr.mean()), float(arr.std()))
        print(f"  Stage {STAGE_NAMES[idx]}: mean={stats[idx][0]:.1f} DOY, std={stats[idx][1]:.1f} DOY  (n={len(arr):,})")

    return stats


def generate_random_predictions(split, data_percentage, stats, seed=42):
    """Iterate over split, sample predictions from per-stage Gaussian, return DataFrame."""
    if split == "test":
        data_split = "testing"
        mask_key = "test"
    elif split == "val":
        data_split = "validation"
        mask_key = "train"  # val tiles use train paper masks
    else:
        data_split = "training"
        mask_key = "train"

    data_paths = get_data_paths(data_split, data_percentage)
    paper_masks = get_masks_paper(mask_key)

    rng = np.random.default_rng(seed)

    rows = []
    for image_paths, gt_path, tile_name in tqdm(data_paths, desc=f"Generating random baseline ({split})"):
        if gt_path is None:
            continue
        if tile_name not in paper_masks:
            continue

        gt = load_raster_output(gt_path)
        gt = gt[CORRECT_INDICES, :, :]  # (4, H, W)

        # Process GT same as CycleDataset.process_gt but keep in DOY
        # Invalid pixels: 32767 or negative
        gt = gt.astype(np.float64)
        invalid = (gt == 32767) | (gt < 0)

        mask = paper_masks[tile_name].cpu().numpy()  # (330, 330) bool
        rs, cs = np.where(mask)

        year, siteid, hlstile = tile_name.split("_")

        for r, c in zip(rs, cs):
            row_data = {
                "index": len(rows),
                "years": int(year),
                "HLStile": hlstile,
                "SiteID": siteid,
                "row": int(r),
                "col": int(c),
                "version": "v1",
            }

            for idx, stage in enumerate(STAGE_NAMES):
                mean, std = stats[idx]
                pred_doy = rng.normal(mean, std)
                # Clamp to valid DOY range
                pred_doy = np.clip(pred_doy, 1, 547)
                row_data[f"{stage}_pred_DOY"] = pred_doy

                # Truth: match model eval convention (process_gt divides by 547,
                # then CSV writer multiplies back by 547; invalid → -1 * 547 = -547)
                if invalid[idx, r, c]:
                    truth_doy = -547.0
                else:
                    truth_doy = float(gt[idx, r, c])
                row_data[f"{stage}_truth_DOY"] = truth_doy

            rows.append(row_data)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Random baseline evaluation for phenology")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--data_percentage", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    out_name = f"random_baseline_{args.data_percentage}_{args.split}.csv"
    out_path = os.path.join(output_dir, out_name)

    if os.path.exists(out_path):
        print(f"Output already exists: {out_path}")
        return

    print("Step 1: Collecting per-stage DOY statistics from training GT ...")
    stats = collect_training_stats(args.data_percentage)

    print(f"\nStep 2: Generating random predictions for {args.split} split ...")
    df = generate_random_predictions(args.split, args.data_percentage, stats, seed=args.seed)

    df.to_csv(out_path, index=False)
    print(f"\nSaved {len(df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
