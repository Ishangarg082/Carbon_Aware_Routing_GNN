#!/bin/bash
# NS-3 + NetAnim Installation Script for WSL
# Run this in WSL Ubuntu to set up ns-3 with Python bindings and NetAnim

set -e

echo "=========================================="
echo "  NS-3 + NetAnim Installation for WSL"
echo "=========================================="
echo ""

# Check if running in WSL
if ! grep -qi microsoft /proc/version; then
    echo "❌ Not running in WSL!"
    echo "Please run this script in WSL Ubuntu"
    exit 1
fi

echo "✓ Running in WSL"
echo ""

# Update system
echo "[1/6] Updating system packages..."
sudo apt update
sudo apt install -y build-essential git python3 python3-dev python3-pip \
    cmake ninja-build ccache libgsl-dev libsqlite3-dev \
    qt5-default qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools

echo ""
echo "[2/6] Installing Python dependencies..."
pip3 install --user cppyy torch torch-geometric networkx matplotlib pandas pyyaml scipy

echo ""
echo "[3/6] Downloading ns-3..."
cd /tmp
if [ -d "ns-3-dev" ]; then
    echo "ns-3-dev already exists, removing..."
    rm -rf ns-3-dev
fi

git clone https://gitlab.com/nsnam/ns-3-dev.git
cd ns-3-dev

echo ""
echo "[4/6] Configuring ns-3 with Python bindings..."
./ns3 configure --enable-python-bindings --enable-examples --enable-tests

echo ""
echo "[5/6] Building ns-3 (this takes 10-15 minutes)..."
./ns3 build

echo ""
echo "[6/6] Building NetAnim..."
cd src/netanim
qmake NetAnim.pro
make

echo ""
echo "=========================================="
echo "  ✓ Installation Complete!"
echo "=========================================="
echo ""
echo "NS-3 installed at: /tmp/ns-3-dev"
echo "NetAnim binary: /tmp/ns-3-dev/src/netanim/NetAnim"
echo ""
echo "To use ns-3 Python bindings, add to ~/.bashrc:"
echo "  export PYTHONPATH=/tmp/ns-3-dev/build/lib:\$PYTHONPATH"
echo ""
echo "To test installation:"
echo "  source ~/.bashrc"
echo "  python3 -c 'import ns.core; print(\"ns-3 loaded!\")'"
echo ""
echo "Next steps:"
echo "  1. Run: source ~/.bashrc"
echo "  2. Copy project to WSL: /mnt/e/nnd_implementation/Carbon_aware_routing_via_GNN_model"
echo "  3. Run: python3 run_ns3_demo.py"
echo ""
