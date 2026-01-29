import yaml
from dotenv import load_dotenv
import os
import re
import shutil
import glob
from pathlib import Path
from loguru import logger
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
# import boto3

import rasterio
from rasterio.mask import mask
from shapely.geometry import box, mapping
import geopandas as gpd

from tqdm import tqdm
from eopf.common.constants import OpeningMode
from eopf.common.file_utils import AnyPath
from eopf.store.convert import convert

from scipy.ndimage import generate_binary_structure, label, binary_fill_holes, binary_erosion
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pyproj import Transformer


######################## ENVIRONMENT SETUP ##########################################################

load_dotenv()

def load_config(config_path="cfg/config.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def setup_environment(config):
    """Set up environment variables and directories for the dataset"""
    # Keep these from environment variables
    ACCESS_KEY_ID = os.environ.get("ACCESS_KEY_ID")
    SECRET_ACCESS_KEY = os.environ.get("SECRET_ACCESS_KEY")
    PAT = os.environ.get("PAT")

    # Get other parameters from config
    ENDPOINT_URL = config['endpoint_url']
    ENDPOINT_STAC = config['endpoint_stac']
    BUCKET_NAME = config['bucket_name']
    DATASET_VERSION = config['dataset_version']
    BASE_DIR = config['base_dir']
    DATASET_DIR = os.path.join(BASE_DIR, DATASET_VERSION)
    BANDS = config['bands']

    # Setup connector
    connector = S3Connector(
        endpoint_url=ENDPOINT_URL,
        access_key_id=ACCESS_KEY_ID,
        secret_access_key=SECRET_ACCESS_KEY,
        region_name='default')

    s3 = connector.get_s3_resource()
    s3_client = connector.get_s3_client()
    bucket = s3.Bucket(BUCKET_NAME)

    return {
        'ACCESS_KEY_ID': ACCESS_KEY_ID,
        'SECRET_ACCESS_KEY': SECRET_ACCESS_KEY,
        'PAT': PAT,
        'ENDPOINT_URL': ENDPOINT_URL,
        'ENDPOINT_STAC': ENDPOINT_STAC,
        'BUCKET_NAME': BUCKET_NAME,
        'DATASET_VERSION': DATASET_VERSION,
        'BASE_DIR': BASE_DIR,
        'DATASET_DIR': DATASET_DIR,
        'BANDS': BANDS,
        's3': s3,
        's3_client': s3_client,
        'bucket': bucket
    }


def prepare_paths(path_dir):
    """
    Prepare paths for input and output datasets from CSV files.

    Args:
        path_dir (str): Directory containing input and target CSV files.

    Returns:
        DataFrame, DataFrame: Two DataFrames for input and output datasets.
    """
    df_input = pd.read_csv(f"{path_dir}/input.csv")
    df_output = pd.read_csv(f"{path_dir}/target.csv")

    df_input["path"] = df_input["Name"].apply(
        lambda x: os.path.join(path_dir, "input", os.path.basename(x).replace(".SAFE", ""))
    )
    df_output["path"] = df_output["Name"].apply(
        lambda x: os.path.join(path_dir, "target", os.path.basename(x).replace(".SAFE", ""))
    )

    return df_input, df_output


###################### S3 CONNECTOR ##############################################################

class S3Connector:
    """A clean connector for S3-compatible storage services"""

    def __init__(self, endpoint_url, access_key_id,
                 secret_access_key, region_name='default'):
        """Initialize the S3Connector with connection parameters"""
        self.endpoint_url = endpoint_url
        self.access_key_id = access_key_id
        self.secret_access_key = secret_access_key
        self.region_name = region_name

        # Create session
        self.session = boto3.session.Session()

        # Initialize S3 resource
        self.s3 = self.session.resource(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name
        )

        # Initialize S3 client
        self.s3_client = self.session.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            region_name=self.region_name
        )

    def get_s3_client(self):
        """Get the boto3 S3 client"""
        return self.s3_client

    def get_s3_resource(self):
        """Get the boto3 S3 resource"""
        return self.s3

    def get_bucket(self, bucket_name):
        """Get a specific bucket by name"""
        return self.s3.Bucket(bucket_name)

    def list_buckets(self):
        """List all available buckets"""
        response = self.s3_client.list_buckets()
        if 'Buckets' in response:
            return [bucket['Name'] for bucket in response['Buckets']]
        return []


###################### ACCESS DATA ##############################################################

def download_sentinel_data(df_output, base_dir, access_key, secret_key, endpoint_url):
    """Download Sentinel data from S3 to local directories"""

    output_dir = os.path.join(base_dir, "target")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Created local directory: {output_dir}")

    S3_CONFIG = {
        "key": access_key,
        "secret": secret_key,
        "client_kwargs": {
            "endpoint_url": endpoint_url,
            "region_name": "default"
        }
    }

    target_store_config = dict(mode=OpeningMode.CREATE_OVERWRITE)

    logger.info("Starting target file downloads...")
    for _, row in tqdm(df_output.iterrows(), total=len(df_output), desc="Target files"):
        try:
            product_url_base = row['S3Path']
            product_url = "s3://" + product_url_base.lstrip("/")
            zarr_filename = os.path.basename(product_url).replace('.SAFE', '.zarr')
            zarr_path = os.path.join(output_dir, zarr_filename)

            # Skip if file already exists
            if os.path.exists(zarr_path):
                logger.info(f"Skipping download, .zarr already exists: {zarr_path}")
                continue
            
            logger.info(f"Downloading target: {product_url} -> {zarr_path}")
            convert(AnyPath(product_url, **S3_CONFIG), zarr_path, target_store_kwargs=target_store_config)
        except Exception as e:
            logger.error(f"Error downloading {product_url if product_url else 'unknown URL'}: {e}")
        finally:
            # Cleanup SAFE temp dirs left in /tmp
            tmp_safes = glob.glob("/tmp/*.SAFE")
            for safe_dir in tmp_safes:
                try:
                    shutil.rmtree(safe_dir, ignore_errors=True)
                    logger.info(f"Cleaned up temporary SAFE: {safe_dir}")
                except Exception as ce:
                    logger.warning(f"Could not clean temp SAFE {safe_dir}: {ce}")


def extract_s2_keys(name):
    """
    Extract (datetime, tile_id) from Sentinel-2 product name
    """
    dt_match = re.search(r"\d{8}T\d{6}", name)
    tile_match = re.search(r"T\d{2}[A-Z]{3}", name)

    if dt_match and tile_match:
        return dt_match.group(), tile_match.group()
    return None, None


def find_gt_png_for_tile(zarr_path, gt_dir):
    zarr_name = os.path.basename(zarr_path)

    zarr_dt, zarr_tile = extract_s2_keys(zarr_name)

    if zarr_dt is None or zarr_tile is None:
        raise ValueError(f"Cannot parse keys from {zarr_name}")

    for f in os.listdir(gt_dir):
        if not f.endswith(".png"):
            continue

        gt_dt, gt_tile = extract_s2_keys(f)

        if gt_dt == zarr_dt and gt_tile == zarr_tile:
            return os.path.join(gt_dir, f)

    raise FileNotFoundError(
        f"No GT PNG found for datetime={zarr_dt}, tile={zarr_tile}"
    )



####################### PATCHES ##################################################################

def pixel_to_world(col, row, transform):
    x = transform.c + col * transform.a + transform.a / 2
    y = transform.f + row * transform.e + transform.e / 2
    return x, y

# Convert from map coords → pixel indices
def world_to_pixel(x, y, transform):
    col = (x - transform.c) / transform.a
    row = (y - transform.f) / transform.e
    return col, row


def load_era5_sst(ds, bbox, target_res, ref_band, date, pat):
    """
    Load ERA5 SST from EDH, crop to bbox, and reproject to Sentinel-2 grid.
    """
    era5 = xr.open_dataset(
        f"https://edh:{pat}@data.earthdatahub.destine.eu/era5/reanalysis-era5-single-levels-v0.zarr",
        chunks={},
        engine="zarr",
    )

    if "sst" not in era5:
        raise ValueError("ERA5 SST variable not found in dataset.")

    sst = era5["sst"]

    # Time selection (closest timestamp)
    if date is not None:
        t = np.datetime64(datetime.fromisoformat(str(date)).replace(tzinfo=None))
        sst = sst.sel(valid_time=t, method="nearest")

    # Robust bbox cropping — ERA5 grid is coarse (~0.25°)
    lon_min, lat_min, lon_max, lat_max = bbox

    if sst.longitude.max() > 180:
        # ERA5 0–360° convention → ensure bbox in same range
        lon_min = lon_min % 360
        lon_max = lon_max % 360

    sst = sst.sel(
        longitude=slice(lon_min - 0.5, lon_max + 0.5),
        latitude=slice(lat_max + 0.5, lat_min - 0.5)  # descending latitude
    )

    if sst.longitude.size == 0 or sst.latitude.size == 0:
        raise ValueError("No ERA5 SST data found in or near the requested bbox.")

    # Rename dims to match rioxarray convention
    sst = sst.rename({"longitude": "x", "latitude": "y"}).transpose("y", "x")

    # Assign CRS (ERA5 is in EPSG:4326)
    sst = sst.rio.write_crs("EPSG:4326")

    # Get reference Sentinel-2 band (for reprojection and shape match)
    crs = ds.attrs.get("other_metadata", {}).get("horizontal_CRS_code")
    ref = ds[f"measurements/reflectance/{target_res}/{ref_band}"].rio.write_crs(crs)

    # Reproject and match resolution to Sentinel-2
    try:
        sst_matched = sst.rio.reproject_match(ref)
    except Exception as e:
        raise RuntimeError(f"Failed to reproject ERA5 SST to Sentinel-2 grid: {e}")

    # Fill missing values and scale if needed (ERA5 SST is in Kelvin)
    sst_matched = sst_matched - 273.15  # convert to °C
    sst_matched = sst_matched.fillna(sst_matched.mean())

    # Match shape exactly (if any small diff remains)
    sst_matched = sst_matched.interp_like(ref, method="nearest")
    print('Min/Max SST after reprojection:', float(sst_matched.min().values), float(sst_matched.max().values))

    return sst_matched


def clean_water_mask(ds, target_res="r10m"):
    """
    Fix water mask by:
    1. Keeping only the sea (remove lakes/rivers).
    2. Filling cloud holes in the sea.
    """
    scl = resample_band(ds, 'scl', target_res=target_res, ref='b04')
    scl = scl.squeeze().values
    raw_water_mask = (scl == 6)
    H, W = raw_water_mask.shape

    # Keep only the largest connected component that touches border (most prob the sea)
    st = generate_binary_structure(2, 2)   # 8-connectivity
    lab, nlab = label(raw_water_mask, structure=st) # different label for each connected water body
    
    if nlab == 0:
        return np.zeros_like(raw_water_mask, dtype=bool)

    # Find component sizes
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # background, we don't consider it

    # Keep the largest component
    largest_label = sizes.argmax()
    sea_only = (lab == largest_label)

    # Fill holes inside sea (caused by clouds)
    sea_filled = binary_fill_holes(sea_only)

    return sea_filled.astype(bool)

def resample_band(ds, band, target_res="r10m", ref="b04", crs="EPSG:32632"):
    """
    Resample any band (reflectance or classification) to target resolution.
    """
    crs = ds.attrs.get("other_metadata", {}).get("horizontal_CRS_code")
    ref_band = ds[f"measurements/reflectance/{target_res}/{ref}"].rio.write_crs(crs) # Reference band at target resolution

    if band == "scl":
        band_da = ds[f"conditions/mask/l2a_classification/r20m/{band}"].rio.write_crs(crs)
        source_res = "r20m"
    else:
        # Detect which reflectance resolution contains the band
        source_res = next(
        (r for r in ["r10m", "r20m", "r60m"] if band in ds[f"measurements/reflectance/{r}"]),
        None
        )
        if source_res is None:
            raise ValueError(f"Band {band} not found in reflectance or scl folder")
        band_da = ds[f"measurements/reflectance/{source_res}/{band}"].rio.write_crs(crs)
    # If source == target, no resampling needed
    if source_res == target_res:
        return band_da
    
    return band_da.rio.reproject_match(ref_band)
    

def build_stack(ds, bands, target_res="r10m", ref_band="b04", crs="EPSG:32632", bbox=None, date=None, pat=None):
    """
    Build a lazy dask-backed (H, W, C) stack from bands, resampling as needed.

    Args:
        ds: xarray Dataset or DataTree
        bands: list of band names to include
        target_res: desired output resolution for all bands
        ref_band: reference band for resampling (default: 'b04' red)
        crs: CRS to assign if missing

    Returns:
        xarray.DataArray with dimensions (y, x, band)
    """
    stack = []

    crs = ds.attrs.get("other_metadata", {}).get("horizontal_CRS_code")

    for b in bands:
        if b in ds['measurements/reflectance/r10m'] or \
           b in ds['measurements/reflectance/r20m'] or \
           b in ds['measurements/reflectance/r60m']:
            arr = resample_band(ds, b, target_res=target_res, ref=ref_band, crs=crs) / 10000.0
        elif b == "sst":
            arr = load_era5_sst(ds, bbox=bbox, target_res=target_res, ref_band=ref_band, date=date, pat=pat)
            #arr = resample_band(ds, b, target_res=target_res, ref=ref_band, crs=crs) / 10000.0
        else:
            raise ValueError(f"Band {b} not found or not supported.")

        # Expand dims for stacking
        arr = arr.expand_dims(band=[b])
        stack.append(arr)

    # Concatenate all bands along 'band' dimension
    print(f"Stacking total of {len(stack)} bands.")
    stacked = xr.concat(stack, dim="band").transpose("y", "x", "band")
    return stacked



def create_patches_dataframe(zarr_files, bands, bbox, target_res, stride, patch_size=256, date=None, pat=None):
    """
    For each zarr file, extract top-left coordinates of sampled patches,
    and store them along with zarr path in a DataFrame.
    
    Returns:
        df_patches: DataFrame with columns ['zarr_path', 'x', 'y']
    """
    patches_per_zarr = {}
    patch_coords = []

    for zf in zarr_files:
        print(f"Processing {zf} for patch coordinates...")
        ds = xr.open_datatree(zf, engine="zarr", mask_and_scale=False, chunks={})
        stack = build_stack(ds, bands,  target_res=target_res, ref_band="b04", bbox=bbox, date=date, pat=pat)

        # Compute water mask
        water_mask = clean_water_mask(ds, target_res=target_res)
        buffered_mask = water_mask & binary_erosion(water_mask, iterations=200)
        valid_mask = buffered_mask.astype(bool)
        
        # Retrieve shape bands
        band = ds['measurements/reflectance/r10m/b04']
        crs_utm = ds.attrs.get("other_metadata", {}).get("horizontal_CRS_code")
        transform = band.rio.transform()
        x_coords = band['x'].values
        y_coords = band['y'].values

        # Convert bbox (EPSG:4326) → tile CRS
        if bbox is not None:
            lon_min, lat_min, lon_max, lat_max = bbox
            transformer = Transformer.from_crs("EPSG:4326", crs_utm, always_xy=True)
            x_min, y_min = transformer.transform(lon_min, lat_min)
            x_max, y_max = transformer.transform(lon_max, lat_max)

            x0, y1 = world_to_pixel(x_min, y_min, transform)  # bottom-left
            x1, y0 = world_to_pixel(x_max, y_max, transform)  # top-right

            # Ensure ordering (y0 < y1, x0 < x1)
            x0, x1 = sorted([int(x0), int(x1)])
            y0, y1 = sorted([int(y0), int(y1)])

            # Clip indices to valid range
            H, W = valid_mask.shape
            x0, x1 = np.clip([x0, x1], 0, W)
            y0, y1 = np.clip([y0, y1], 0, H)

            valid_mask = valid_mask[y0:y1, x0:x1]
            stack = stack.isel(x=slice(x0, x1), y=slice(y0, y1))

        # Convert to numpy for patch extraction
        stack_np = stack.transpose('y', 'x', 'band').values
        H, W, _ = stack_np.shape

        print(f"AOI shape for {zf}: {H}x{W} pixels")

        patches_per_zarr[zf] = []
        count = 0

        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patch_mask = valid_mask[y:y+patch_size, x:x+patch_size]

                # Skip patches mostly outside water
                if patch_mask.mean() < 0.70:
                    print(f"Skipping patch at x={x+x0}, y={y+y0} due to low water coverage ({patch_mask.mean():.2f})")
                    continue

                patch = stack_np[y:y+patch_size, x:x+patch_size, :]
                if np.isnan(patch).any() or np.isinf(patch).any():
                    patch = np.nan_to_num(patch, nan=np.nanmean(patch), posinf=1.0, neginf=0.0)
                patches_per_zarr[zf].append(patch)

                patch_coords.append({
                    "zarr_file": zf,
                    "x_pix": x + x0,
                    "y_pix": y + y0
                })
                count += 1
    
    df_coords = pd.DataFrame(patch_coords)
    print(f"Total patches collected: {sum(len(patches) for patches in patches_per_zarr.values())}")

    return patches_per_zarr, df_coords


def export_geotiff_and_vector(zarr_path, prob_map, binary_mask, confidence=None, amei=None, out_dir="results"):
    ds = xr.open_datatree(zarr_path, engine="zarr", mask_and_scale=False)
    ref_band = ds["measurements/reflectance/r10m/b04"]
    transform = ref_band.rio.transform()
    crs = ds.attrs.get("other_metadata", {}).get("horizontal_CRS_code")

    # stack available layers
    bands = [prob_map.astype(np.float32), binary_mask.astype(np.uint8)]
    if confidence is not None:
        bands.append(confidence.astype(np.float32))
    if amei is not None:
        bands.append(amei.astype(np.float32))
    arr = np.stack(bands, axis=0)

    # zarr name
    zarr_name = os.path.basename(zarr_path).replace('.zarr', '')

    out_path = os.path.join(out_dir, f"mucilage_mask.tif") # ADAPTED filename: mucilage_mask_{zarr_name}.tif
    # out_path = os.path.join(out_dir, f"mucilage_mask.tif") old version for unique .tif output

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=arr.dtype.name,
        crs=crs,
        transform=transform,
        compress="lzw"
    ) as dst:
        for i, band in enumerate(arr, 1):
            dst.write(band, i)
    print("Saved:", out_path)
    return out_path


def crop_tiff_to_bbox(tif_path, bbox, out_path):
    """
    Crop GeoTIFF to a bbox given in EPSG:4326 (lat/lon).
    The function will automatically reproject bbox to match raster CRS.
    """
    with rasterio.open(tif_path) as src:
        raster_crs = src.crs
        print(f"Raster CRS: {raster_crs}")
        
        # Create GeoDataFrame from bbox in EPSG:4326
        bbox_geom = gpd.GeoDataFrame(
            {"geometry": [box(*bbox)]},
            crs="EPSG:4326"
        )
        
        # Reproject bbox to raster CRS
        bbox_geom = bbox_geom.to_crs(raster_crs)
        geom = [mapping(bbox_geom.iloc[0].geometry)]
        
        # Perform crop
        out_image, out_transform = mask(src, geom, crop=True)
        out_meta = src.meta.copy()
        
        out_meta.update({
            "driver": "GTiff",
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"✅ Cropped image saved to {out_path}")


def evaluate_patchwise(
    df_coords,
    preds_list,
    gt_mask,
    zarr_path,
    patch_size=256
):
    df_tile = df_coords[df_coords["zarr_file"] == zarr_path].reset_index(drop=True)

    y_true_all = []
    y_pred_all = []

    gt_mucilage_pixel_count = 0
    gt_total_pixel_count = 0
    n_patches_used = 0

    for i, row in df_tile.iterrows():
        y = int(row["y"])
        x = int(row["x"])

        gt_patch = gt_mask[y:y+patch_size, x:x+patch_size]
        pred_patch = preds_list[i]

        # Safety checks
        if gt_patch.shape != pred_patch.shape:
            continue
        if gt_patch.size == 0:
            continue

        y_true_all.append(gt_patch.flatten())
        y_pred_all.append(pred_patch.flatten())

        gt_mucilage_pixel_count += int(gt_patch.sum())
        gt_total_pixel_count += gt_patch.size
        n_patches_used += 1

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n_pixels_evaluated": int(y_true.size),
        "n_patches_used": n_patches_used,
        "gt_mucilage_pixels": gt_mucilage_pixel_count,
        "gt_mucilage_fraction": gt_mucilage_pixel_count / max(gt_total_pixel_count, 1),
    }