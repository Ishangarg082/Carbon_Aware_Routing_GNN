# Carbon-Aware Routing via GNN Model

This project implements carbon-aware network routing using a combination of an MLP carbon predictor and a Graph Attention Network (GAT). The idea is simple: instead of always picking the shortest path, we predict the carbon cost of different routes and pick the one that emits the least CO2.

We train a model on real network routing data (`carbon_network_data.csv`), and validate the approach using NS-3 network simulation with NetAnim visualization.

## Results

- Model R² score: **0.88** on held-out test data
- Average carbon reduction: **~13.6%** over shortest-path baseline
- Up to **28.4%** reduction on multi-hop flows
- Model is lightweight — only 11,649 parameters

## How to Run

### Prerequisites

- Python 3.8 or higher
- PyTorch 2.0+
- For NS-3 simulation: WSL (Ubuntu) with ns-3.48 installed

### Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Train the model

The pre-trained weights are already included (`models/best_carbon_predictor.pth`), but if you want to retrain:

```bash
python training/train_real_model.py
```

This reads from `carbon_network_data.csv`, trains the MLP, and saves the weights and scaler to `models/`.

### Run the standalone demo

```bash
python demo_carbon_routing.py
```

This runs everything locally without NS-3. It creates a random network, generates traffic, and compares three routing strategies:
1. Baseline (OSPF shortest path)
2. Threshold-based carbon avoidance (avoids top-25% dirtiest nodes)
3. Our GNN/MLP carbon-aware routing

Results get saved to `results/` — plots, CSV metrics, and a markdown report.

### Run with NS-3 (full simulation)

You need NS-3 installed in WSL for this. There's a setup script included:

```bash
wsl
cd /mnt/e/nnd_implementation/Carbon_aware_routing_via_GNN_model
bash setup_ns3_wsl.sh
```

Then set up the environment variables (or add them to `~/.bashrc`):

```bash
export PATH=~/ns-allinone-3.48/ns-3.48/ns3-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PYTHONPATH=~/ns-allinone-3.48/ns-3.48/build/bindings/python:$PYTHONPATH
export LD_LIBRARY_PATH=~/ns-allinone-3.48/ns-3.48/build/lib:$LD_LIBRARY_PATH
```

Install dependencies in WSL:

```bash
source ~/ns-allinone-3.48/ns-3.48/ns3-venv/bin/activate
pip install -r requirements_wsl.txt
```

Run the simulation:

```bash
bash run_ns3_wsl.sh
```

You can customize it:

```bash
bash run_ns3_wsl.sh --hours 6 --nodes 10
bash run_ns3_wsl.sh --hours 48 --nodes 30 --netanim
bash run_ns3_wsl.sh --no-netanim
```

### Run Multi-Seed Experiment (Statistically Valid Results)

For paper-quality results, we run the simulation multiple times with different random seeds to eliminate noise and compute valid statistics (Cohen's d, paired t-tests, win-rates).

You can run this either with the standalone Python script (faster, no NS-3 dependencies) or via the full NS-3 simulation in WSL:

**1. Standalone (Faster, Python only):**
```bash
# Recommended full run (15 seeds, 48 hours, 200 nodes)
python run_multi_seed_experiment.py

# Quick smoke test (5 seeds, 12 hours, 50 nodes)
python run_multi_seed_experiment.py --seeds 5 --hours 12 --nodes 50 --results-dir results/smoke_test
```

**2. Full NS-3 Simulation (In WSL):**
Multi-seed experiments are run by default after the main simulation when using the full NS-3 simulation. You can customize the number of seeds, or disable it completely.

```bash
# Runs a full 15-seed NS-3 simulation (multi-seed is enabled by default)
bash run_ns3_wsl.sh --hours 48 --nodes 200

# Customize the number of seeds
bash run_ns3_wsl.sh --seeds 20

# Disable multi-seed experiment
bash run_ns3_wsl.sh --no-multi-seed
```

**Expected Results:**
Both methods will generate files in the `results/` folder, including:
- `multi_seed_report.md`: The main statistical analysis report
- `ms_01_boxplot_total_carbon.png`: Box plot of total carbon per run
- `ms_02_errorbar_mean_ci.png`: Mean carbon with 95% confidence intervals
- `ms_04_stats_summary_panel.png`: A text panel summarizing the key metrics

*(Note: The `results/smoke_test/` folder contains output from the smaller test run. You should use the files directly in `results/` for the full 15-seed simulation.)*

### Run tests

```bash
python tests/test_gnn_model.py
```

## Project Structure

```
├── carbon_network_data.csv         Training dataset (6000 samples)
├── enhanced_gnn_model.py           GAT model with temporal encoding
├── demo_carbon_routing.py          Standalone demo (no NS-3 needed)
├── run_ns3_demo.py                 NS-3 simulation entry point
├── run_ns3_wsl.sh                  WSL launch script
├── setup_ns3_wsl.sh                NS-3 installation helper
├── requirements.txt                Python deps
├── requirements_wsl.txt            WSL deps (includes cppyy)
│
├── models/
│   ├── carbon_predictor.py         MLP model definition
│   ├── best_carbon_predictor.pth   Trained weights
│   └── feature_scaler.pkl          Feature normalization
│
├── training/
│   ├── real_data_loader.py         Loads and splits the CSV data
│   └── train_real_model.py         Training script
│
├── simulation/
│   ├── network_topology.py         Builds the network graph
│   ├── carbon_profiles.py          Time-varying carbon intensity
│   ├── energy_model.py             Per-node energy model
│   ├── traffic_matrix.py           Traffic generation
│   └── gnn_routing_controller.py   GNN routing logic for NS-3
│
├── controllers/
│   └── mlp_routing_controller.py   MLP routing controller
│
├── integration/
│   ├── ns3_bindings.py             NS-3 Python bindings wrapper
│   └── netanim_helper.py           NetAnim utilities
│
├── visualization/
│   ├── metrics_analyzer.py         Metrics and reporting
│   ├── dashboard.py                Plot generation
│   └── generate_animation.py       HTML animation
│
├── config/
│   ├── simulation_config.yaml
│   └── training_config.yaml
│
├── tests/
│   └── test_gnn_model.py
│
└── results/                        Generated at runtime
```

## How it works

There are two models in this project:

**CarbonPredictor (MLP)** — the simpler model used for standalone routing. Takes 9 features as input (hop count, packet/byte counts, flow duration, CPU usage, carbon intensity, and protocol type) and predicts the carbon emission for that route. We train it on `carbon_network_data.csv`.

**CarbonAwareGAT** — the graph neural network used in NS-3 simulation. It uses multi-head graph attention (4 heads, 3 layers) with temporal encoding so it's aware of time-of-day patterns in carbon intensity. It predicts per-edge link weights that the simulation uses to update OSPF routing tables.

The routing controller finds candidate paths between source and destination, runs each through the predictor, and picks the path with the lowest predicted carbon.

## Dataset

`carbon_network_data.csv` contains 6,000 routing flow samples with the following columns:

| Column | Description |
|--------|-------------|
| num_hops | Number of hops in the path |
| packet_count | Packets in the flow |
| byte_count | Bytes transferred |
| flow_duration | Duration in ms |
| cpu_usage | CPU utilization (%) |
| carbon_intensity | Grid carbon intensity (gCO2/kWh) |
| protocol | TCP, UDP, or ICMP |
| carbon_emission | Target — actual carbon emitted (gCO2eq) |

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`** — make sure the venv is activated before running anything.

**Model file not found** — run `python training/train_real_model.py` to generate the weights.

**NS-3 not loading in WSL** — check that the environment variables are set correctly. You can verify with: `python3 -c "from ns import ns; print('ok')"`.

**cppyy freezes in WSL** — this happens when Windows paths leak into `$PATH`. Fix it by resetting PATH: `export PATH=~/ns3-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`
