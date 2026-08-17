#!/bin/bash
# NS-3 + NetAnim Installation Script for WSL (corrected)
# Run this in WSL Ubuntu to set up ns-3 with Python bindings and NetAnim
#
# Fixes vs original:
#   - Installs into $HOME (persists across WSL restarts), not /tmp
#   - Uses the stable ns-3.48 release tarball, not the bleeding-edge dev branch
#   - Uses a Python venv instead of pip3 --user (avoids externally-managed-environment error)
#   - Drops qt5-default (removed in newer Ubuntu; not actually needed)
#   - Disables the nr (5G) contrib module by default to keep build times/RAM sane
#   - Skips re-cloning/re-downloading/re-building if ns-3 already exists (safe to re-run)

set -e

echo "=========================================="
echo "  NS-3 + NetAnim Installation for WSL"
echo "=========================================="
echo ""

# Check if running in WSL
if ! grep -qi microsoft /proc/version; then
    echo "Not running in WSL!"
    echo "Please run this script in WSL Ubuntu"
    exit 1
fi

echo "Running in WSL"
echo ""

NS3_VERSION="3.48"
INSTALL_DIR="$HOME/ns-allinone-${NS3_VERSION}"
NS3_DIR="${INSTALL_DIR}/ns-${NS3_VERSION}"
VENV_DIR="${NS3_DIR}/ns3-venv"

# ---- [1/6] System packages ----
echo "[1/6] Updating system packages..."
sudo apt update
sudo apt install -y build-essential git python3 python3-dev python3-venv python3-pip \
    cmake ninja-build ccache libgsl-dev libsqlite3-dev \
    qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools

# ---- [2/6] Python venv + dependencies ----
echo ""
echo "[2/6] Setting up Python virtual environment and dependencies..."

if [ ! -d "$NS3_DIR" ]; then
    mkdir -p "$INSTALL_DIR"
fi

# Download and extract ns-3 first if not already present, so the venv can live inside NS3_DIR
if [ ! -d "$NS3_DIR" ]; then
    echo "  ns-3 not yet downloaded; will fetch in step 3 before creating venv."
fi

# ---- [3/6] Download ns-3 (stable release, not dev branch) ----
echo ""
echo "[3/6] Downloading ns-3 ${NS3_VERSION} (stable release)..."
if [ -d "$NS3_DIR" ]; then
    echo "  ${NS3_DIR} already exists, skipping download/extract."
else
    cd "$HOME"
    wget "https://www.nsnam.org/releases/ns-allinone-${NS3_VERSION}.tar.bz2"
    tar xjf "ns-allinone-${NS3_VERSION}.tar.bz2"
fi

cd "$NS3_DIR"

# Now create the venv inside the ns-3 directory
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip
pip install cppyy torch torch-geometric networkx matplotlib pandas pyyaml scipy

# ---- [4/6] Configure ns-3 with Python bindings ----
echo ""
echo "[4/6] Configuring ns-3 with Python bindings..."
./ns3 configure --enable-python-bindings --enable-examples --enable-tests --disable-modules=nr

# ---- [5/6] Build ns-3 ----
echo ""
echo "[5/6] Building ns-3 (this can take a while; -j2 keeps memory use in check on WSL)..."
./ns3 build -j2

# ---- [6/6] Build NetAnim ----
echo ""
echo "[6/6] Building NetAnim..."
cd "${NS3_DIR}/src/netanim" 2>/dev/null || cd "${NS3_DIR}/netanim" 2>/dev/null || {
    echo "  NetAnim source directory not found at expected paths; skipping."
    echo "  You can build it manually later if needed."
}
if [ -f "NetAnim.pro" ]; then
    qmake NetAnim.pro
    make
fi

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "NS-3 installed at: ${NS3_DIR}"
echo "Python venv at:    ${VENV_DIR}"
echo "NetAnim binary (if built): ${NS3_DIR}/src/netanim/NetAnim"
echo ""
echo "To use ns-3 Python bindings in a new terminal:"
echo "  source ${VENV_DIR}/bin/activate"
echo ""
echo "To test installation:"
echo "  source ${VENV_DIR}/bin/activate"
echo "  python3 -c 'from ns import ns; print(\"ns-3 loaded!\")'"
echo ""
echo "Next steps:"
echo "  1. source ${VENV_DIR}/bin/activate"
echo "  2. cd /mnt/e/nnd_implementation/Carbon_aware_routing_via_GNN_model"
echo "  3. python3 run_ns3_demo.py"
echo ""