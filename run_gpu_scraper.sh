#!/bin/bash

echo "🚀 Setting up GPU-accelerated Bing News Scraper..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
else
    echo "⚠️  NVIDIA GPU not detected, will fall back to CPU processing"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install Playwright browsers
echo "🌐 Installing Playwright browsers..."
playwright install chromium

# Run the GPU-accelerated scraper
echo "🔍 Starting GPU-accelerated concurrent scraping..."
python bing_scraper_gpu.py

echo "✅ Scraping completed! Check bing_news_articles_gpu.csv for results." 