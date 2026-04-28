# agentic-recommenders

Agentic recommenders for MovieLens

## Overview

Implementation inspired by

## Architecture & Components

- **Core package:** `agentic_rec/`
  - `data.py`: Data loading and preprocessing (MovieLens, LanceDB)

## Installation

Requirements

- Python 3.12+ (the project is developed and tested on 3.12)
- The repository uses `uv` to manage virtual environments and tasks.
  See `pyproject.toml` for pinned dependencies.

Install dependencies with uv (recommended):

```bash
# set up the environment and install pinned deps
uv sync
```

## Usage

### Data preparation

This repo ships helper scripts to download and convert MovieLens
datasets into Parquet and LanceDB formats.

Example: prepare MovieLens 1M (ml-1m) and write parquet files into
`data/`:

```bash
# fetch, extract and convert to parquet
uv run data
```

If you already have the original files (for example `ml-1m.zip`), place
them under `data/` and `uv run data` will pick them up. Otherwise the
script will download and extract the dataset.

## Development notes & troubleshooting

- If you see dependency or Python version errors, confirm you are using
  Python 3.12 and run `uv sync` to recreate the virtual environment.

## References

- [ARAG: Agentic Retrieval Augmented Generation for Personalized Recommendation][arag]

[arag]: https://arxiv.org/abs/2506.21931
