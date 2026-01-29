import glob
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import zipfile
#from sklearn.model_selection import train_test_split

from utils.utils import *
from utils.cdse_utils import *
from utils.torch import define_model, load_model_weights
from utils.plot import *

import traceback
import sys

# Show DeltaTwin logs
WORKDIR = os.environ.get("DELTA_WORKDIR", "/tmp")
LOG_FILE = os.path.join(WORKDIR, "flood_detector.log")

# Redirect stdout/stderr to log file (optional, but recommended)
sys.stdout = open(LOG_FILE, "a", buffering=1)
sys.stderr = open(LOG_FILE, "a", buffering=1)


def main():

    # Setup
    args = parser.parse_args()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(BASE_DIR, "cfg", "config.yaml")
    config = load_config(config_path=config_path)
    DATASET_DIR = os.path.join(BASE_DIR, config['dataset_version'])
    OUTPUT_DIR = os.getenv("DELTA_OUTPUT_DIR", "outputs")
    today = datetime.utcnow().date()

# Create Dataset
    # Parameters from dataset config
    logger.info("Starting dataset creation...")
    query_config = config['query']
    bands = config['bands']
    bbox = args.bbox
    start_date = today if args.start_date.lower() == "today" else datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = (today + timedelta(days=1)) if args.end_date.lower() == "today" else datetime.strptime(args.end_date, "%Y-%m-%d")
    mid_date = start_date + (end_date - start_date) / 2
    max_items = query_config['max_items']
    max_cloud_cover = query_config['max_cloud_cover']

    print(f"Querying data from {start_date} to {end_date} for bbox {bbox}...")

    all_l2a_results = query_sentinel_data(
        bbox, start_date, end_date, max_items, max_cloud_cover
    )

    # Process and align data
    df_l2a = queries_curation(all_l2a_results)
    #df_l2a.to_csv(f"{DATASET_DIR}/output_l2a.csv")

    logger.info("Starting download process...")

    df_l2a = df_l2a.drop([0])
    download_sentinel_data(
        df_output = df_l2a,
        base_dir = DATASET_DIR,
        access_key = args.cdse_key,
        secret_key = args.cdse_secret,
        endpoint_url = 'https://eodata.dataspace.copernicus.eu'
    )

    logger.success("All downloads completed.")


# Patchify
    logger.info("Extracting patch coordinates...")
    zarr_dir = os.path.join(DATASET_DIR, "target")
    zarr_files = glob.glob(os.path.join(zarr_dir, "*.zarr"))

    if not zarr_files:
        logger.warning(f"No Zarr files found in {zarr_dir}")
        return

    patches_per_zarr, df_coords = create_patches_dataframe(
        zarr_files,
        bands=bands,
        bbox=bbox,
        target_res='r10m',
        stride=256,
        patch_size=config['download'].get('patch_size', 256),
        date=mid_date,
        pat=args.earth_data_hub_pat
    )
    #np.save(os.path.join(DATASET_DIR, 'patches.npy'), patches)
    logger.success("Patch extraction completed.")

    #visualize_sst(patches[100], save_dir="results")

    # visualize_patches_on_tile(
    #     zarr_file=zarr_files[0],
    #     patches_coords=df_coords[df_coords["zarr_file"] == zarr_files[0]],
    #     patch_size=256,
    #     bbox=config["query"]["bbox"],
    #     save_dir=os.path.join(BASE_DIR, "results")
    # )

# Segmentation
    # Parameters from model config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model checkpoints
    checkpoint = os.path.join(BASE_DIR, "weights/unet_checkpoint.pth")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    # Load config
    config = ckpt['config']
    mean = ckpt['mean']
    std = ckpt['std']

    # Load model
    model = load_model_weights(config, ckpt, device)

    # visualize_patch_prediction(patch, probs, pred, save_dir="results", patch_id=f"patch_{i}")

    df_coords = df_coords.rename(columns={"x_pix": "x", "y_pix": "y"})

    # Group by tile
    unique_zarrs = df_coords['zarr_file'].unique()

    tif_paths = []

    for zarr_path in unique_zarrs:

        all_probs_per_patch = []
        all_preds_per_patch = []

        patches = patches_per_zarr[zarr_path] 

        for i, patch in enumerate(patches):
            patch = (patch - mean) / (std + 1e-8)
            patch_tensor = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).float().to(device)

            model.eval()
            with torch.no_grad():
                logits = model(patch_tensor)
                if logits.shape[1] == 1:
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
                else:
                    probs = torch.softmax(logits, dim=1)[:, 1, :, :].squeeze().cpu().numpy()

                pred = (probs > 0.5).astype(np.uint8)
                
            all_probs_per_patch.append(probs)
            all_preds_per_patch.append(pred)

        avg_prob, binary_mask = stitch_predictions_old(
            zarr_file=zarr_path,
            df_coords=df_coords,
            probs_list=all_probs_per_patch,
            preds_list=all_preds_per_patch,
            patch_size=256
        )

        #output_dir = os.path.join(DATASET_DIR, "outputs")
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_dir = OUTPUT_DIR

        tif_path = export_geotiff_and_vector(
            zarr_path=zarr_path,
            prob_map=avg_prob,
            binary_mask=binary_mask,
            confidence=None,   # or your own metric
            amei=None,
            out_dir=BASE_DIR
        )

        crop_tiff_to_bbox(tif_path, args.bbox, os.path.join(BASE_DIR, tif_path))
        tif_paths.append(tif_path)

        # # Create ZIP of all TIFs
        # zip_path = os.path.join(BASE_DIR, "mucilage_masks.zip")
        # with zipfile.ZipFile(zip_path, 'w') as zipf:
        #     for tif in tif_paths:
        #         zipf.write(tif, os.path.basename(tif))

        # visualize_final_panel(
        #     zarr_path=zarr_path,
        #     avg_prob=avg_prob,
        #     binary_mask=binary_mask,
        #     df_coords=df_coords.assign(patch_size=256),
        #     out_path=os.path.join("results", "final_panel.png")
        # )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single patch")
    parser.add_argument("--cdse_key", type=str, required=True)
    parser.add_argument("--cdse_secret", type=str, required=True)
    parser.add_argument("--earth_data_hub_pat", type=str, required=True)
    #parser.add_argument("--bbox", type=str, help="Bounding box [minx miny maxx maxy]")
    parser.add_argument("--bbox", type=float, nargs=4, help="Bounding box [minx miny maxx maxy]")
    parser.add_argument("--start_date", type=str, required=False, help="Start date in 'YYYY-MM-DD'. 'today' if current date.")
    parser.add_argument("--end_date", type=str, required=False, help="End date in 'YYYY-MM-DD'. 'today' if current date.")
    try:
    # Entire main logic here
        main()  # whatever is in your main()
    except Exception as e:
        with open(LOG_FILE, "w") as f:
            f.write("Exception:\n")
            traceback.print_exc(file=f)
        raise 