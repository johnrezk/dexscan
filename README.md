# dexscan

**To see a more recent example of my coding style and best practices, take a look at my Python caching library [rapide 🐎](https://github.com/johnrezk/rapide)**

This was a personal project I worked on between June - October 2023,
to further hone my skills in data processing, interacting with the Ethereum network, and machine learning.

The initial idea was to scrape per-block data on all holders of any given on-chain crypto asset. Each per-block dataset would contain accurate statistics on live holder profit-and-losses, along with other data points. This type of data was not offered by any third-party data brokers at the time, and would have cost a fortune to acquire through managed nodes.

## Features

- ✅ Efficiently query and cache data from a single, self-hosted Ethereum node
- ✅ Per-block, per-address live profit and loss data, fast enough for live use cases
- ✅ Use minimal 3rd party data sources
- ✅ Prepare normalized feature set from large amounts of raw data
- ✅ Train both neural-network and gradient boosting models
- ✅ UI that displays live model inference on a given crypto asset
- ✅ Simple backtesting system for models
- ✅ Uses EdgeDB, use EdgeQL to create concise queries
- ✅ Runs even on consumer hardware

## What I Would Change

Looking back on this project, there is a number of improvements and simplifications I would make.

- **Rework Data Features:** At the time of this project I didn't have an understanding of predictive financial models, and simply included all the data I had collected. I compensated with regularization in the training steps, but this was a band-aid. Making a profitable model was not really my focus at the time.

- **Switch to SQLite:** EdgeDB was fun to use, but introduces too much overhead and complexity given the scope of this project. A simple SQLite implementation for the cached data would probably be faster and give fewer headaches, especially when writing many rows in a single transaction. Some of the tables would have to be redesigned to better fit the more traditional style of SQL queries.

- **Testing for Faster Iteration:** This codebase originally grew out of a small single script. Certain components, such as the `Database` and `Hybrid` classes, are fairly complex. I would add a suite of unit and integration tests, especially for these components, before making any further changes. This would be essential if switching to SQLite.

## Requirements to Run

- Python 3.11
- [poetry](https://python-poetry.org/docs/) for package management
- [edgedb-cli](https://docs.edgedb.com/get-started/quickstart) to setup database
- A fully synced, archival, self-hosted Ethereum node
  - I recommend [reth](https://github.com/paradigmxyz/reth) for execution and [lighthouse](https://github.com/sigp/lighthouse) for consensus

## Command Reference

Used to interact with the library

### App - Start Realtime UI

```bash
poetry run ui
```

This prompts for a token pair address, then loads a terminal UI displaying live data and charts for that pair, including PnL spread amongst addresses and model inference results.

### Sampling - Collect Samples

```bash
poetry run collect-samples
```

Using the pair addresses provided in `samples_test.csv` and `samples_train.csv`, this script pulls all per-block data from the database or node as needed, generates feature set, then packages into a compressed JSON for training.

### Sampling - Reset Pair

```bash
poetry run reset-pair
```

Prompts for a token pair address and clears associated data from the database

### Training - Run NN Training

```bash
poetry run train
```

Begins model training process. Each epoch is saved in a project directory.

### Training - Save NN Epoch

```bash
poetry run train-save-epoch
```

Prompts for a epoch number in the training output directory and saves the model.

### Training - Run LGBM Training

```bash
poetry run train-lgbm
```

Begins LGBM training process.

### Backtest - Run

```bash
poetry run backtest
```

One of the last things I experimented with. Runs a basic backtesting suite and outputs results to JSON file.

### Backtest - Analyze Results

```bash
poetry run backtest-analyze
```

Prompts for backtest results file and outputs some useful interpretive statistics in the terminal.
