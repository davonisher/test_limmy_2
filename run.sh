#!/bin/bash

# Jobs Scraper Runner Script
# Voor gebruik op Ubuntu server via Limmy

echo "Starting Jobs Scraper..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python 3.7+ first."
    exit 1
fi

# Check if requirements are installed
if [ ! -f "requirements.txt" ]; then
    echo "requirements.txt not found!"
    exit 1
fi

# Install dependencies if needed
echo "Installing dependencies..."
pip3 install -r requirements.txt

# Run the scraper
echo "Running jobs scraper..."
python3 jobs_scraper_limmy.py

echo "Scraper finished!" 