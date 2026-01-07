import os
import pandas as pd
import numpy as np
import torch.nn.functional as F
from pyproj import Transformer
import xarray as xr

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
matplotlib.use("Agg")

from utils.utils import resample_band


def get_rgb(patch):
    """
    Convert multi-band patch to RGB for visualization.
    bands: tuple with indices for (R, G, B).
    """
    rgb = patch[:, :, [3, 2, 1]] #now there is b1 as well
    p2, p98 = np.nanpercentile(rgb, (2, 98))
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-6), 0, 1)
    
    return rgb


def show_amei(patch, eps=1e-6, ax=None):
    green = patch[:, :, 2]
    red = patch[:,:,3]
    nir = patch[:,:,7]
    swir = patch[:,:,10]
    denom = green + 0.25 * swir
    amei  = (2*red + nir - 2*swir) / (denom + eps)

    p2, p98 = np.nanpercentile(amei, (2, 98))
    amei = np.clip((amei - p2) / (p98 - p2), 0, 1)
    cmap = plt.cm.turbo

    if ax is None:
        ax = plt.gca()
    ax.imshow(amei, cmap=cmap)
    ax.axis("off")
    return ax


def visualize_patches_on_tile(zarr_file, patches_coords, patch_size=256, bbox=None, target_res="r10m", save_dir=None):
    """
    Visualize the RGB tile with overlayed red rectangles representing patch positions.

    Args:
        zarr_file (str): Path to the Zarr file
        patches_coords (pd.DataFrame): Output of create_patches_dataframe (filtered for this zarr)
        patch_size (int): Size of each patch in pixels
        bbox (list): Optional [lon_min, lat_min, lon_max, lat_max]
        target_res (str): Resolution to read (usually 'r10m')
    """
    print(f"Visualizing patches on {zarr_file}...")

    # --- Load RGB bands ---
    ds = xr.open_datatree(zarr_file, engine="zarr", mask_and_scale=False, chunks={})
    b04 = ds[f"measurements/reflectance/{target_res}/b04"].values  # Red
    b03 = ds[f"measurements/reflectance/{target_res}/b03"].values  # Green
    b02 = ds[f"measurements/reflectance/{target_res}/b02"].values  # Blue
    rgb = np.stack([b04, b03, b02], axis=-1)
    
    # Normalize for display
    rgb = np.clip(rgb / np.nanpercentile(rgb, 99), 0, 1)
    H, W, _ = rgb.shape
    print(f"Tile shape: {H}x{W}")

    # --- Prepare the figure ---
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(rgb, origin="upper")

    # --- Overlay patches ---
    for _, row in patches_coords.iterrows():
        rect = mpatches.Rectangle(
            (row["x_pix"], row["y_pix"]),  # pixel top-left corner
            patch_size,
            patch_size,
            linewidth=1,
            edgecolor="red",
            facecolor="none"
        )
        ax.add_patch(rect)

    ax.set_title(f"Patches overlay")
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")

    # overlay bbox outline
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
        x_min, y_min = transformer.transform(lon_min, lat_min)
        x_max, y_max = transformer.transform(lon_max, lat_max)

        # Convert world to pixel coords
        band = ds[f"measurements/reflectance/{target_res}/b04"]
        transform = band.rio.transform()

        def world_to_pixel(x, y, transform):
            col = (x - transform.c) / transform.a
            row = (transform.f - y) / -transform.e
            return col, row

        x0, y1 = world_to_pixel(x_min, y_min, transform)
        x1, y0 = world_to_pixel(x_max, y_max, transform)
        rect = mpatches.Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
            linestyle="--"
        )
        ax.add_patch(rect)
        ax.text(x0, y0 - 20, "BBox", color="yellow")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"patches_on_tile_2.png"), dpi=200)
    plt.close(fig)


def visualize_sst(patch, save_dir):
    sst = patch[:, :, -1]

    plt.figure(figsize=(5, 5))
    im = plt.imshow(sst, cmap="coolwarm", vmin=5, vmax=35)
    plt.colorbar(im, label="SST (°C)")
    plt.title(f"SST Band")
    plt.savefig(os.path.join(save_dir, f"sst.png"), dpi=200)
    plt.close()


def visualize_patch_prediction(patch, probs, pred, save_dir, patch_id="patch"):
    os.makedirs(save_dir, exist_ok=True)

    # Get RGB for visualization
    rgb = get_rgb(patch)  # typically b04,b03,b02 or your chosen combination
    mask = pred.astype(bool)

    # Overlay mask in red
    red_mask = np.zeros_like(rgb)
    red_mask[..., 0] = 1.0
    alpha = 0.5
    overlay = np.where(mask[..., None], (1 - alpha) * rgb + alpha * red_mask, rgb)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Input RGB")
    axes[0].axis("off")

    axes[1].imshow(probs, cmap="viridis")
    axes[1].set_title("Predicted Probability")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay (Red = Mucilage)")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{patch_id}_prediction.png"), dpi=200)
    plt.close(fig)


# ---------- Stitch / aggregation helpers ----------
def stitch_predictions(zarr_file, df_coords, means_list, vars_list=None, patch_size=256, prob_threshold=0.5):
    """
    Rebuild full-tile mosaics from patch-level predictions.

    Parameters
    ----------
    zarr_file : str
        Path to the Zarr tile.
    df_coords : pd.DataFrame
        Patch coordinates with columns ['zarr_file', 'x', 'y'].
    means_list : list[np.ndarray]
        List of patch-level probability means (H, W).
    vars_list : list[np.ndarray] or None
        List of patch-level variances (H, W). If None, uncertainty is not computed.
    patch_size : int
        Patch size in pixels.
    prob_threshold : float
        Threshold for binary mask generation.

    Returns
    -------
    mean_tile : np.ndarray
        Tile-level mean probability map.
    std_tile : np.ndarray or None
        Tile-level uncertainty map (None if vars_list is None).
    binary_mask : np.ndarray
        Tile-level binary mask.
    """

    # Open reference tile for shape
    ds = xr.open_datatree(zarr_file, engine="zarr", mask_and_scale=False)
    ref = ds["measurements/reflectance/r10m/b04"]
    H, W = ref.shape

    # Accumulators
    mean_sum = np.zeros((H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.uint16)

    if vars_list is not None:
        second_moment_sum = np.zeros((H, W), dtype=np.float64)

    # Select patch coordinates for this tile
    df_coords_zarr = (
        df_coords[df_coords["zarr_file"] == zarr_file]
        .reset_index(drop=True)
    )

    # Stitch patches
    for i, row in df_coords_zarr.iterrows():
        top, left = int(row["y"]), int(row["x"])
        mean_patch = means_list[i]
        h, w = mean_patch.shape

        mean_sum[top:top+h, left:left+w] += mean_patch
        count[top:top+h, left:left+w] += 1

        if vars_list is not None:
            var_patch = vars_list[i]
            second_moment_sum[top:top+h, left:left+w] += mean_patch**2 + var_patch

    # Compute tile-level mean
    mean_tile = np.divide(mean_sum, count, where=count > 0)

    # Compute uncertainty if available
    if vars_list is not None:
        second_moment_tile = np.divide(
            second_moment_sum, count, where=count > 0
        )
        var_tile = second_moment_tile - mean_tile**2
        var_tile = np.maximum(var_tile, 0.0)  # numerical safety
        std_tile = np.sqrt(var_tile)
    else:
        std_tile = None

    # Binary mask (always produced)
    binary_mask = (mean_tile >= prob_threshold).astype(np.uint8)

    return mean_tile, std_tile, binary_mask


def stitch_predictions_old(zarr_file, df_coords, probs_list, preds_list, patch_size=256):
    """Rebuild full-tile probability and binary mosaics from patch-level results - NO STD"""
    ds = xr.open_datatree(zarr_file, engine="zarr", mask_and_scale=False)
    ref = ds["measurements/reflectance/r10m/b04"]
    H, W = ref.shape
    prob_sum = np.zeros((H, W), np.float32)
    count = np.zeros((H, W), np.uint16)

    # Stitch patch probabilities
    df_coords_zarr = df_coords[df_coords['zarr_file'] == zarr_file].reset_index(drop=True)
    for i, row in df_coords_zarr.iterrows():
        top, left = int(row["y"]), int(row["x"])
        prob = probs_list[i]
        h, w = prob.shape
        prob_sum[top:top+h, left:left+w] += prob
        count[top:top+h, left:left+w] += 1

    avg_prob = np.divide(prob_sum, count, where=count > 0)
    binary_mask = (avg_prob >= 0.5).astype(np.uint8)
    return avg_prob, binary_mask


def visualize_final_panel(zarr_path, avg_prob, binary_mask, avg_uncert, df_coords, out_path="results/final_panel.png"):
    """
    Create a 3-panel visualization:
      1) RGB
      2) RGB + probability overlay
      3) RGB + red mask + patch contours
      4) RGB + uncertainty heatmap
    """

    # Lazy loading zarr 
    ds = xr.open_zarr(os.path.join(zarr_path, "measurements/reflectance/r10m"), consolidated=False)
    b04 = ds["b04"].values
    b03 = ds["b03"].values
    b02 = ds["b02"].values
    rgb = np.stack([b04, b03, b02], axis=-1).astype(np.float32)
    rgb = rgb.astype(np.float32)
    rgb = np.clip(rgb / np.percentile(rgb, 98), 0, 1)

    prob = np.nan_to_num(avg_prob, 0).astype(np.float32)
    uncert = np.nan_to_num(avg_uncert, 0).astype(np.float32)
    mask = (binary_mask > 0).astype(np.uint8)

    # Normalize uncertainty for visualization only
    uncert_norm = uncert / (np.percentile(uncert, 99) + 1e-8)
    uncert_norm = np.clip(uncert_norm, 0, 1)

    # # Draw patch boundaries
    # patch_outline = np.zeros_like(mask)
    # patch_size = df_coords.iloc[0].get("patch_size", 256)

    # for _, row in df_coords.iterrows():
    #     top, left = int(row["y"]), int(row["x"])
    #     patch_outline[top:top+patch_size, left] = 1
    #     patch_outline[top:top+patch_size, left+patch_size-1] = 1
    #     patch_outline[top, left:left+patch_size] = 1
    #     patch_outline[top+patch_size-1, left:left+patch_size] = 1

    # fig, axs = plt.subplots(1, 4, figsize=(24, 8))

    # # 1) RGB
    # axs[0].imshow(rgb)
    # axs[0].set_title("RGB Tile")
    # axs[0].axis("off")

    # # 2) RGB + probability
    # axs[1].imshow(rgb)
    # axs[1].imshow(prob, cmap="viridis", alpha=0.6)
    # axs[1].set_title("Probability Overlay")
    # axs[1].axis("off")

    # # 3) RGB + mask
    # overlay = rgb.copy()
    # overlay[mask.astype(bool)] = (
    #     0.4 * rgb[mask.astype(bool)] + 0.6 * np.array([1, 0, 0])
    # )
    # overlay[patch_outline > 0] = [0, 0, 0]
    # axs[2].imshow(overlay)
    # axs[2].set_title("Mask + Patch Contours")
    # axs[2].axis("off")

    # # 4) RGB + uncertainty
    # axs[3].imshow(rgb)
    # axs[3].imshow(uncert_norm, cmap="magma", alpha=0.6)     
    # axs[3].set_title("Epistemic Uncertainty")
    # axs[3].axis("off")

    # plt.tight_layout()
    # plt.savefig(out_path, dpi=200)
    # plt.close()


# ---------- Visual helpers ----------
def get_rgb_from_tile(ds, bands=('b04','b03','b02'), target_res='r10m'):
    """
    Build an RGB array matching the sentinel-2 ref grid (target_res).
    Uses resample_band or build_stack approach: return array HxWx3 in range [0,1].
    ds: opened xarray datatree/dataset for the tile.
    """
    # Use your resample_band function to get each band at target_res, then stack
    arrs = []
    for b in bands:
        # band_da will be xarray with dims (y,x)
        band_da = resample_band(ds, b, target_res=target_res, ref='b04', crs="EPSG:32633")
        band_np = band_da.values.astype(np.float32)
        arrs.append(band_np)
    # stack into H, W, 3
    rgb = np.dstack([arrs[0], arrs[1], arrs[2]])
    # simple percentile scaling per-band (2-98)
    for i in range(3):
        p2, p98 = np.nanpercentile(rgb[..., i], (2, 98))
        rgb[..., i] = np.clip((rgb[..., i] - p2) / (p98 - p2 + 1e-8), 0, 1)
    return rgb