#!/bin/bash

set -e

echo "🐳 Building Docker container..."
docker-compose build

echo "✅ Setup complete!"
echo ""
echo "To start using the project:"
echo "  1. Start container: docker-compose up -d"
echo "  2. Download data: docker-compose exec sleep-classifier bash download_data.sh"
echo "  3. Run training: docker-compose exec sleep-classifier python train_kfold.py"
