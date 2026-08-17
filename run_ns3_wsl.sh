#!/bin/bash
# =============================================================================
# Carbon-Aware Routing - NS-3 WSL Launch Script
#
# This script sets up the correct environment and runs the ns-3 simulation.
# Run this from WSL:
#   cd /mnt/e/nnd_implementation/Carbon_aware_routing_via_GNN_model
#   bash run_ns3_wsl.sh
#
# Or with options:
#   bash run_ns3_wsl.sh --hours 6 --nodes 10
#   bash run_ns3_wsl.sh --no-netanim
# =============================================================================

set -e

echo "============================================"
echo " Carbon-Aware Routing - NS-3 Simulation"
echo "============================================"
echo ""

# --- Environment Setup ---

# 1. Activate virtual environment
VENV_PATH="$HOME/ns-allinone-3.48/ns-3.48/ns3-venv"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✓ Virtual environment activated"
else
    echo "⚠ ns3-venv not found at $VENV_PATH"
    echo "  Create it using setup_ns3_wsl.sh"
    exit 1
fi

# 2. Strip Windows paths from PATH (prevents cppyy freeze in WSL)
export PATH=$VENV_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
echo "✓ PATH cleaned (Windows paths removed)"

# 3. Set ns-3 Python bindings path
NS3_DIR="$HOME/ns-allinone-3.48/ns-3.48"
if [ -d "$NS3_DIR" ]; then
    export PYTHONPATH="$NS3_DIR/build/bindings/python:$PYTHONPATH"
    export LD_LIBRARY_PATH="$NS3_DIR/build/lib:$LD_LIBRARY_PATH"
    echo "✓ ns-3.48 paths configured"
else
    echo "❌ ns-3 not found at $NS3_DIR"
    exit 1
fi

# 4. Navigate to project directory
PROJECT_DIR="/mnt/e/nnd_implementation/Carbon_aware_routing_via_GNN_model"
if [ -d "$PROJECT_DIR" ]; then
    cd "$PROJECT_DIR"
    echo "✓ Working directory: $PROJECT_DIR"
else
    echo "❌ Project not found at $PROJECT_DIR"
    exit 1
fi

# 5. Quick sanity check
echo ""
echo "Verifying ns-3 Python bindings..."
python3 -c "from ns import ns; print('✓ ns-3 bindings loaded successfully')" || {
    echo "❌ Failed to load ns-3 bindings"
    echo "  Try rebuilding: cd $NS3_DIR && ./ns3 configure --enable-python-bindings && ./ns3 build"
    exit 1
}

# --- Run Simulation ---
echo ""
echo "Starting simulation..."
echo "--------------------------------------------"
python3 run_ns3_demo.py "$@"
