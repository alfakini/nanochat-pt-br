"""
Repackage downloaded Carolina corpus XML files to parquet shards and upload to HuggingFace.

Usage:
    # Repackage from downloaded files
    python dev/repackage_carolina.py --input dataset/raw/ --output dataset/

    # Upload existing parquet shards
    python dev/repackage_carolina.py --upload --input dataset/

The nanochat/dataset.py is configured to use the local dataset/ directory.
"""

import os
import glob
import gzip
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterator

import pyarrow.parquet as pq
import pyarrow as pa
from huggingface_hub import HfApi

HF_UPLOAD_REPO = "alfakini/carolina-corpus-shuffled"
CHARS_PER_SHARD = 250_000_000
ROW_GROUP_SIZE = 1024

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def parse_gzipped_xml(xml_path: str) -> Iterator[str]:
    """Parse a gzipped XML file, yielding text content from each document."""
    with gzip.open(xml_path, "rt", encoding="utf-8") as f:
        content = f.read()

    root = ET.fromstring(content)
    ns = "http://www.tei-c.org/ns/1.0"

    for text_elem in root.findall(f".//{{{ns}}}text"):
        body = text_elem.find(f"{{{ns}}}body")
        if body is not None:
            for p in body.findall(f".//{{{ns}}}p"):
                if p.text and len(p.text.strip()) > 50:
                    yield p.text.strip()


def find_xml_files(input_dir: str) -> list[str]:
    """Find all .xml.gz files in the input directory."""
    pattern = os.path.join(input_dir, "**", "*.xml.gz")
    return sorted(glob.glob(pattern, recursive=True))


def repackage(
    input_dir: str,
    output_dir: str,
    upload: bool = False,
):
    """Repackage XML files to parquet shards."""
    os.makedirs(output_dir, exist_ok=True)

    xml_files = find_xml_files(input_dir)
    print(f"Found {len(xml_files)} XML files in {input_dir}")

    total_docs = 0
    total_chars = 0
    shard_index = 0
    shard_docs = []
    shard_chars = 0

    t0 = time.time()

    for file_idx, xml_path in enumerate(xml_files, 1):
        filename = os.path.basename(xml_path)
        print(f"[{file_idx}/{len(xml_files)}] Processing {filename}...")

        try:
            for text in parse_gzipped_xml(xml_path):
                if len(text) < 50:
                    continue

                shard_docs.append(text)
                shard_chars += len(text)
                total_docs += 1
                total_chars += len(text)

                if shard_chars >= CHARS_PER_SHARD and len(shard_docs) >= ROW_GROUP_SIZE:
                    shard_path = os.path.join(
                        output_dir, f"shard_{shard_index:05d}.parquet"
                    )
                    _write_shard(shard_path, shard_docs)

                    dt = time.time() - t0
                    print(
                        f"  Wrote shard_{shard_index:05d}.parquet | #docs: {len(shard_docs)} | #chars: {shard_chars} | time: {dt:.1f}s"
                    )

                    shard_docs = []
                    shard_chars = 0
                    shard_index += 1
        except Exception as e:
            print(f"  Error: {e}")

        if file_idx % 100 == 0:
            print(
                f"\n  Progress: {file_idx}/{len(xml_files)} files | {total_docs} docs | {shard_index} shards\n"
            )

    if shard_docs:
        shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
        _write_shard(shard_path, shard_docs)
        print(
            f"Wrote final shard: shard_{shard_index:05d}.parquet | #docs: {len(shard_docs)}"
        )

    total_time = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Repackaging complete!")
    print(f"Total documents: {total_docs:,}")
    print(f"Total characters: {total_chars:,}")
    print(f"Total shards: {shard_index + 1}")
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Output: {output_dir}")

    if upload:
        print(f"\nUploading to HuggingFace: {HF_UPLOAD_REPO}")
        api = HfApi(token=os.getenv("HF_TOKEN"))
        api.upload_large_folder(
            folder_path=output_dir,
            repo_id=HF_UPLOAD_REPO,
            repo_type="dataset",
        )
        print("Upload complete!")


def upload_only(output_dir: str):
    """Upload existing parquet shards to HuggingFace."""
    shards = sorted([f for f in os.listdir(output_dir) if f.endswith(".parquet")])
    print(f"Found {len(shards)} parquet shards in {output_dir}")
    print(f"Uploading to {HF_UPLOAD_REPO}...")

    api = HfApi(token=os.getenv("HF_TOKEN"))
    api.upload_large_folder(
        folder_path=output_dir,
        repo_id=HF_UPLOAD_REPO,
        repo_type="dataset",
    )
    print("Upload complete!")


def _write_shard(shard_path: str, docs: list[str]):
    """Write a shard to parquet file."""
    shard_table = pa.Table.from_pydict({"text": docs})
    pq.write_table(
        shard_table,
        shard_path,
        row_group_size=ROW_GROUP_SIZE,
        use_dictionary=False,
        compression="zstd",
        compression_level=3,
        write_statistics=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Repackage Carolina corpus to parquet shards"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input directory containing downloaded XML files (or parquet shards if using --upload)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset",
        help="Output directory for parquet shards (default: dataset)",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload converted shards to HuggingFace",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output)

    if args.upload:
        input_dir = args.input or output_dir
        if not os.path.exists(input_dir):
            print(f"Error: Directory {input_dir} does not exist")
            exit(1)
        upload_only(input_dir)
    else:
        if not args.input:
            print("Error: --input is required for repackaging")
            exit(1)
        input_dir = os.path.abspath(args.input)
        repackage(input_dir, output_dir, upload=args.upload)
