"""
Download Corpus Carolina XML files from HuggingFace.

The Carolina corpus is stored as XML files organized by taxonomy:
- wikis, datasets_and_other_corpora, legislative_branch, judicial_branch,
- university_domains, social_media, public_domain_works

Usage:
    python dev/download_carolina.py --output dataset/raw/

This will download all ~554 XML.gz files (~3GB total) to the output directory.
"""

import os
import time
import requests
from huggingface_hub import HfApi
from concurrent.futures import ThreadPoolExecutor, as_completed

CORPUS_CAROLINA_REPO = "carolina-c4ai/corpus-carolina"
CORPUS_CAROLINA_BASE = (
    "https://huggingface.co/datasets/carolina-c4ai/corpus-carolina/resolve/main"
)
CORPUS_DIR = "corpus"
TAXONOMIES = [
    "datasets_and_other_corpora",
    "judicial_branch",
    "legislative_branch",
    "public_domain_works",
    "social_media",
    "university_domains",
    "wikis",
]


def get_xml_urls() -> list[str]:
    """Get list of URLs for all XML files using HuggingFace API."""
    print("Getting file list from HuggingFace...")
    api = HfApi()
    all_files = api.list_repo_files(repo_id=CORPUS_CAROLINA_REPO, repo_type="dataset")

    urls = []
    for taxonomy in TAXONOMIES:
        taxonomy_files = [
            f
            for f in all_files
            if f.startswith(f"{CORPUS_DIR}/{taxonomy}/") and f.endswith(".xml.gz")
        ]
        taxonomy_urls = [f"{CORPUS_CAROLINA_BASE}/{f}" for f in taxonomy_files]
        urls.extend(taxonomy_urls)
        print(f"  {taxonomy}: {len(taxonomy_files)} files")

    print(f"Total: {len(urls)} XML files")
    return urls


def download_file(url: str, output_dir: str) -> tuple[str, bool]:
    """Download a single file maintaining directory structure."""
    relative_path = url.split("/resolve/main/")[1]
    output_path = os.path.join(output_dir, relative_path)

    if os.path.exists(output_path):
        print(f"Skipping (exists): {relative_path}")
        return url, True

    print(f"Downloading: {relative_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    temp_path = output_path + ".tmp"
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()

        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

        os.rename(temp_path, output_path)
        return url, True
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"  Error downloading {relative_path}: {e}")
        return url, False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Carolina corpus XML files")
    parser.add_argument(
        "--output",
        type=str,
        default="dataset/raw",
        help="Output directory for XML files (default: dataset/raw)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers (default: 8)",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    urls = get_xml_urls()
    print(f"\nDownloading to: {output_dir}")
    print(f"Using {args.workers} workers")
    print()

    t0 = time.time()
    successful = 0
    total = len(urls)

    print()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_file, url, output_dir): url for url in urls}

        for future in as_completed(futures):
            url, success = future.result()
            if success:
                successful += 1

    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Download complete!")
    print(f"Successful: {successful}/{total}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Output: {output_dir}")
