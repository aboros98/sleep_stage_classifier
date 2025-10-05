#!/bin/bash

# Sleep Classification Data Download Script
# This script downloads the Sleep-Accel dataset from PhysioNet

set -e  # Exit on any error

echo "🚀 Starting Sleep-Accel dataset download..."

# Configuration
DATA_DIR="./data"
DOWNLOAD_URL="https://physionet.org/files/sleep-accel/1.0.0/"
TEMP_DIR="./temp_download"

# Create data directory if it doesn't exist
echo "📁 Creating data directory..."
mkdir -p "$DATA_DIR"

# Create temporary download directory
mkdir -p "$TEMP_DIR"

# Function to clean up on exit
cleanup() {
    echo "🧹 Cleaning up temporary files..."
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

# Check if wget is installed
if ! command -v wget &> /dev/null; then
    echo "❌ Error: wget is not installed. Please install wget first."
    echo "   On macOS: brew install wget"
    echo "   On Ubuntu/Debian: sudo apt-get install wget"
    echo "   On CentOS/RHEL: sudo yum install wget"
    exit 1
fi

# Download the dataset
echo "⬇️  Downloading Sleep-Accel dataset from PhysioNet..."
echo "   This may take several minutes depending on your internet connection..."

# Use wget to download recursively
# -r: recursive
# -N: turn on time-stamping
# -c: continue partial downloads
# -np: no parent directories
# -P: download to specific directory
wget -r -N -c -np "$DOWNLOAD_URL" -P "$TEMP_DIR"

# Check if download was successful
if [ ! -d "$TEMP_DIR/physionet.org/files/sleep-accel/1.0.0" ]; then
    echo "❌ Error: Download failed or directory structure is unexpected."
    exit 1
fi

echo "📦 Moving files to data directory..."

# Move files from the nested directory structure to our data directory
DOWNLOAD_PATH="$TEMP_DIR/physionet.org/files/sleep-accel/1.0.0"

# Check if the expected directories exist
if [ -d "$DOWNLOAD_PATH" ]; then
    # Move all contents to data directory
    cp -r "$DOWNLOAD_PATH"/* "$DATA_DIR"/
    echo "✅ Successfully moved files to $DATA_DIR"
else
    echo "❌ Error: Expected download structure not found."
    exit 1
fi

# Check if we have the expected directories
EXPECTED_DIRS=("motion" "labels" "clinical")
MISSING_DIRS=()

for dir in "${EXPECTED_DIRS[@]}"; do
    if [ ! -d "$DATA_DIR/$dir" ]; then
        MISSING_DIRS+=("$dir")
    fi
done

if [ ${#MISSING_DIRS[@]} -eq 0 ]; then
    echo "✅ All expected directories found:"
    for dir in "${EXPECTED_DIRS[@]}"; do
        file_count=$(find "$DATA_DIR/$dir" -type f | wc -l)
        echo "   📂 $dir: $file_count files"
    done
else
    echo "⚠️  Warning: Some expected directories are missing:"
    for dir in "${MISSING_DIRS[@]}"; do
        echo "   ❌ $dir"
    done
fi

# Display summary
echo ""
echo "🎉 Dataset download completed!"
echo "📊 Summary:"
echo "   📁 Data directory: $DATA_DIR"
total_size=$(du -sh "$DATA_DIR" | cut -f1)
total_files=$(find "$DATA_DIR" -type f | wc -l)
echo "   📏 Total size: $total_size"
echo "   📄 Total files: $total_files"

echo ""
echo "🚀 You can now start training:"
echo "   python train.py              # Single model training"
echo "   python train_kfold.py        # K-fold cross-validation"  
echo "   python logistic_regression.py # Baseline model"
echo ""
echo "🐳 Or use Docker:"
echo "   docker-compose up dev        # Development mode"
