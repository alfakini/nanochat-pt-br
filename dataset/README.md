# Dataset Directory

This directory contains the Carolina corpus converted to parquet shards for nanochat training.

## Setup

```bash
# Step 1: Download XML files (~3GB)
python dev/download_carolina.py --output dataset/raw/

# Step 2: Repackage to parquet shards
python dev/repackage_carolina.py --input dataset/raw/ --output dataset/
```

## Source

- **Corpus**: [Carolina](https://huggingface.co/datasets/carolina-c4ai/corpus-carolina)
- **Description**: A General Corpus of Contemporary Brazilian Portuguese (~3GB, ~2M documents)
- **License**: CC-BY-4.0
- **Maintainers**: Carolina C4AI team at University of Sao Paulo

## Format

Each shard is a parquet file with:
- Single column: `text` (string)
- Row group size: 1024
- Compression: zstd

## Citation

```bibtex
@misc{carolina2023,
  title={Carolina: a General Corpus of Contemporary Brazilian Portuguese},
  author={Crespo et al.},
  year={2023},
  eprint={2303.16098},
  archivePrefix={arXiv},
  primaryClass={cs.CL}
}
```
