#!/bin/bash

# Product Hunt Scraper Cron Job Runner
# This script starts the scraper in cron mode (runs every 4 hours)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}=== Product Hunt Scraper Cron Job ===${NC}"
echo -e "${YELLOW}Starting scraper in cron mode (every 4 hours)${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop the cron job${NC}"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: python3 is not installed or not in PATH${NC}"
    exit 1
fi

# Check if required files exist
if [ ! -f "enhanced_scraper_cron.py" ]; then
    echo -e "${RED}Error: enhanced_scraper_cron.py not found in current directory${NC}"
    exit 1
fi

if [ ! -f "PH_check1.csv" ]; then
    echo -e "${RED}Error: PH_check1.csv not found in current directory${NC}"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to handle script termination
cleanup() {
    echo -e "\n${YELLOW}Received termination signal. Stopping cron job...${NC}"
    # The Python script handles graceful shutdown internally
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start the cron job
echo -e "${GREEN}Starting cron job...${NC}"
echo -e "${BLUE}Logs will be saved to:${NC}"
echo -e "  - scraper_cron.log (Python logs)"
echo -e "  - logs/cron_runner.log (Shell script logs)"
echo ""

# Run the cron job with logging
python3 enhanced_scraper_cron.py --cron 2>&1 | tee logs/cron_runner.log 