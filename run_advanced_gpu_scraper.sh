#!/bin/bash

echo "🚀 Setting up Advanced GPU-accelerated Bing News Scraper..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
    echo ""
else
    echo "⚠️  NVIDIA GPU not detected, will fall back to CPU processing"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install Playwright browsers
echo "🌐 Installing Playwright browsers..."
playwright install chromium

# Run the advanced GPU-accelerated scraper
echo "🔍 Starting Advanced GPU-accelerated concurrent scraping..."
echo "This will scrape 5 pages per company with GPU acceleration..."
python bing_scraper_gpu_advanced.py

echo "✅ Advanced scraping completed!"
echo "📊 Check the following files for results:"
echo "   - bing_news_articles_gpu_advanced.csv (main data)"
echo "   - bing_news_articles_gpu_advanced.json (JSON format)"
echo "   - scraping_summary.json (statistics)" 