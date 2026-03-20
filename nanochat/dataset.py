"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see:
- `repackage_data_reference.py` (legacy datasets)
- `dev/repackage_carolina.py` (current Carolina corpus)
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq

from nanochat.common import get_base_dir

# -----------------------------------------------------------------------------
# The specifics of the current pretraining dataset

# The Carolina corpus: A General Corpus of Contemporary Brazilian Portuguese
# https://huggingface.co/datasets/carolina-c4ai/corpus-carolina
# Converted to parquet shards using dev/repackage_carolina.py
# Hosted at https://huggingface.co/datasets/alfakini/carolina-corpus-shuffled

HF_REPO = "alfakini/carolina-corpus-shuffled"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
MAX_SHARD = None  # Set dynamically based on files in HuggingFace repo
index_to_filename = lambda index: f"shard_{index:05d}.parquet"
base_dir = get_base_dir()
DATA_DIR = os.path.join(base_dir, "base_data_carolina")

# -----------------------------------------------------------------------------
# These functions are useful utilities to other modules, can/should be imported


_hf_file_list = None  # Cache for HuggingFace file list


def list_parquet_files(data_dir=None, warn_on_legacy=False):
    """Returns full paths to parquet files, downloading from HuggingFace on-demand."""
    global _hf_file_list

    data_dir = DATA_DIR if data_dir is None else data_dir
    os.makedirs(data_dir, exist_ok=True)

    # First check local files
    local_files = sorted(
        [
            f
            for f in os.listdir(data_dir)
            if f.endswith(".parquet") and not f.endswith(".tmp")
        ]
    )

    # If we have local files, return them
    if local_files:
        return [os.path.join(data_dir, f) for f in local_files]

    # Otherwise, fetch file list from HuggingFace
    if _hf_file_list is None:
        from huggingface_hub import HfApi

        api = HfApi()
        print("Fetching file list from HuggingFace...")
        all_files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
        _hf_file_list = sorted([f for f in all_files if f.endswith(".parquet")])
        global MAX_SHARD
        MAX_SHARD = len(_hf_file_list) - 1
        print(f"Found {len(_hf_file_list)} parquet files on HuggingFace")

    return [os.path.join(data_dir, f) for f in _hf_file_list]


def parquets_iter_batched(split, start=0, step=1):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". the last parquet file will be val.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            yield texts


def download_single_file(filename):
    """Downloads a single file from HuggingFace with some backoff."""

    # Construct the local filepath for this file and skip if it already exists
    filepath = os.path.join(DATA_DIR, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filename} (already exists)")
        return True

    # Construct the remote URL for this file
    url = f"{HF_BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    # Download with retries
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            # Write to temporary file first
            temp_path = filepath + f".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
            # Move temp file to final location
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")
            # Clean up any partial files
            for path in [filepath + f".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        pass
            # Try a few times with exponential backoff: 2^attempt seconds
            if attempt < max_attempts:
                wait_time = 2**attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed

    parser = argparse.ArgumentParser(
        description="Download pretraining dataset shards from HuggingFace"
    )
    parser.add_argument(
        "-n",
        "--num-files",
        type=int,
        default=-1,
        help="Number of train shards to download (default: -1 = all). "
        "Validation shard is always downloaded.",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )
    args = parser.parse_args()

    # Fetch file list from HuggingFace
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Fetching file list from {HF_REPO}...")
    all_files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
    parquet_files = sorted([f for f in all_files if f.endswith(".parquet")])
    print(f"Found {len(parquet_files)} parquet files")

    MAX_SHARD = len(parquet_files) - 1

    # Prepare the output directory
    os.makedirs(DATA_DIR, exist_ok=True)

    # Download files
    num_to_download = (
        len(parquet_files)
        if args.num_files == -1
        else min(args.num_files, len(parquet_files) - 1)
    )
    files_to_download = parquet_files[:num_to_download]
    # Always include last file as validation
    if parquet_files[-1] not in files_to_download:
        files_to_download.append(parquet_files[-1])

    print(
        f"Downloading {len(files_to_download)} shards using {args.num_workers} workers..."
    )
    print(f"Target directory: {DATA_DIR}")
    print()

    successful = 0
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(download_single_file, f): f for f in files_to_download
        }
        for future in as_completed(futures):
            if future.result():
                successful += 1

    print(
        f"Done! Downloaded: {successful}/{len(files_to_download)} shards to {DATA_DIR}"
    )
