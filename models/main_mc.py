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

from PIL import Image
import segmentation_models_pytorch as smp

def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout2d):
            m.train()

def add_decoder_dropout(model, p=0.5):
    for module in model.decoder.modules():
        if isinstance(module, smp.base.modules.Conv2dReLU):
            # Append dropout after the ReLU
            module.add_module("dropout", nn.Dropout2d(p))

def load_gt_png(gt_path):
    gt = Image.open(gt_path).convert("L")  # grayscale
    gt = np.array(gt)

    # Ensure binary {0,1}
    if gt.max() > 1:
        gt = (gt > 0).astype(np.uint8)

    return gt


def main():

    # Setup
    args = parser.parse_args()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(BASE_DIR, "cfg", "config.yaml")
    config = load_config(config_path=config_path)
    DATASET_DIR = os.path.join(BASE_DIR, config['dataset_version'])
    OUTPUT_DIR = os.getenv("DELTA_OUTPUT_DIR", "outputs")
    GT_DIR = os.path.join(DATASET_DIR, "mucilage_masks_marmara")
    today = datetime.utcnow().date()

    N_MC = 100
    eps = 1e-8


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

    all_l2a_results = query_sentinel_data(
        bbox, start_date, end_date, max_items, max_cloud_cover
    )

    # Process and align data
    df_l2a = queries_curation(all_l2a_results)

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
    checkpoint = os.path.join(BASE_DIR, "weights/unet_checkpoint_dropout.pth")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    # Load config
    config = ckpt['config']
    mean = ckpt['mean']
    std = ckpt['std']

    # Load model
    model = load_model_weights(config, ckpt, device)
    add_decoder_dropout(model, p=0.5) # added dropout to decoder

    # visualize_patch_prediction(patch, probs, pred, save_dir="results", patch_id=f"patch_{i}")

    df_coords = df_coords.rename(columns={"x_pix": "x", "y_pix": "y"})

    # Group by tile
    unique_zarrs = df_coords['zarr_file'].unique()

    tif_paths = []
    
    for zarr_path in unique_zarrs:
        all_means_per_patch = []
        all_vars_per_patch = []

        patches = patches_per_zarr[zarr_path]

        for patch in patches:
            # Normalize
            patch = (patch - mean) / (std + 1e-8)
            patch_tensor = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).float().to(device)

            # Activate Dropout
            enable_dropout(model)

            mc_probs = []
            with torch.no_grad():
                for _ in range(N_MC):
                    logits = model(patch_tensor)
                    if logits.shape[1] == 1:
                        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
                    else:
                        prob = torch.softmax(logits, dim=1)[:,1,:,:].squeeze().cpu().numpy()

                    mc_probs.append(prob)

            mc_probs = np.stack(mc_probs, axis=0)  # [T,H,W]

            # MC STATISTICS
            mean_patch = mc_probs.mean(axis=0)
            var_patch  = mc_probs.var(axis=0, ddof=1)

            # Collect results
            all_means_per_patch.append(mean_patch)
            all_vars_per_patch.append(var_patch)

        # Stitch per-tile
        mean_tile, std_tile, binary_mask = stitch_predictions(
            zarr_file=zarr_path,
            df_coords=df_coords,
            means_list=all_means_per_patch,
            vars_list=all_vars_per_patch,
            patch_size=256,
            prob_threshold=0.6
        )

        # Tile-level uncertainty indicator (for alarms / reporting)
        positive_mask = binary_mask == 1
        print("Numer of positive pixels:", np.sum(positive_mask))

        if np.any(positive_mask):
            tile_uncertainty_pos = std_tile[positive_mask].mean()
            tile_uncertainty = std_tile[~positive_mask].mean()
            print("Mean uncertainty over positive pixels:", tile_uncertainty_pos)
            print("Mean uncertainty over water pixels:", tile_uncertainty)
        else:
            tile_uncertainty = std_tile.mean()
            print(f"Tile uncertainty: {tile_uncertainty}")


        # # Evaluate masks
        # gt_path = find_gt_png_for_tile(zarr_path, GT_DIR)
        # logger.info(f"Matched GT: {os.path.basename(gt_path)}")

        # gt_mask = load_gt_png(gt_path)

        # metrics = evaluate_patchwise(
        #     df_coords=df_coords,
        #     preds_list=all_preds_per_patch,
        #     gt_mask=gt_mask,
        #     zarr_path=zarr_path,
        #     patch_size=256
        # )

        # logger.success(f"Evaluation for {os.path.basename(zarr_path)}")
        # for k, v in metrics.items():
        #     logger.info(f"{k}: {v}")

        #output_dir = os.path.join(DATASET_DIR, "outputs")

        # os.makedirs(OUTPUT_DIR, exist_ok=True)
        # output_dir = OUTPUT_DIR

        # tif_path = export_geotiff_and_vector(
        #     zarr_path=zarr_path,
        #     prob_map=mean_tile,
        #     binary_mask=binary_mask,
        #     confidence = std_tile,
        #     amei=None,
        #     out_dir=BASE_DIR
        # )

        # crop_tiff_to_bbox(tif_path, args.bbox, tif_path)
        # tif_paths.append(tif_path)

        # # Create ZIP of all TIFs
        # zip_path = os.path.join(BASE_DIR, "mucilage_masks.zip")
        # with zipfile.ZipFile(zip_path, 'w') as zipf:
        #     for tif in tif_paths:
        #         zipf.write(tif, os.path.basename(tif))

        # visualize_final_panel(
        #     zarr_path=zarr_path,
        #     avg_prob=mean_tile,
        #     binary_mask=binary_mask,
        #     avg_uncert=std_tile,
        #     df_coords=df_coords.assign(patch_size=256),
        #     out_path=os.path.join(output_dir, "final_panel.png")
        # )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single patch")
    parser.add_argument("--cdse_key", type=str, required=True)
    parser.add_argument("--cdse_secret", type=str, required=True)
    parser.add_argument("--earth_data_hub_pat", type=str, required=True)
    parser.add_argument("--bbox", type=float, nargs=4, help="Bounding box [minx miny maxx maxy]")
    parser.add_argument("--start_date", type=str, required=False, help="Start date in 'YYYY-MM-DD'. 'today' if current date.")
    parser.add_argument("--end_date", type=str, required=False, help="End date in 'YYYY-MM-DD'. 'today' if current date.")
    main()