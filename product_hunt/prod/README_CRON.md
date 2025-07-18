# Product Hunt Scraper - Cron Job Setup

This enhanced scraper now includes cron job functionality to automatically run every 4 hours.

## Features

- **Automatic Scheduling**: Runs every 4 hours automatically
- **Timestamped Output**: Each run creates a new file with timestamp
- **Graceful Shutdown**: Handles Ctrl+C and system signals properly
- **Comprehensive Logging**: Logs to both file and console
- **Error Recovery**: Automatically retries on errors
- **Flexible Modes**: Can run single scrape or continuous cron job

## Usage

### Option 1: Run Single Scrape (Default)
```bash
python3 enhanced_scraper_cron.py
```

### Option 2: Run Cron Job (Every 4 Hours)
```bash
python3 enhanced_scraper_cron.py --cron
```

### Option 3: Use Shell Script (Recommended)
```bash
./run_cron_scraper.sh
```

## Configuration

Edit the configuration variables at the top of `enhanced_scraper_cron.py`:

```python
# Configuration
TOOLS_CSV_PATH = 'PH_check1.csv'
CRON_INTERVAL_HOURS = 4  # Run every 4 hours
SAMPLE_SIZE = 500
```

## Output Files

- **Scraped Results**: `scraped_results_YYYYMMDD_HHMMSS.csv`
- **Python Logs**: `scraper_cron.log`
- **Shell Logs**: `logs/cron_runner.log` (when using shell script)

## System Service Setup (Optional)

For production deployment, you can run as a system service:

1. Edit the service file:
   ```bash
   nano product-hunt-scraper.service
   ```

2. Update paths and username:
   ```ini
   User=your_username
   WorkingDirectory=/full/path/to/product_hunt/prod
   ExecStart=/usr/bin/python3 /full/path/to/product_hunt/prod/enhanced_scraper_cron.py --cron
   ```

3. Install and start the service:
   ```bash
   sudo cp product-hunt-scraper.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable product-hunt-scraper
   sudo systemctl start product-hunt-scraper
   ```

4. Check status:
   ```bash
   sudo systemctl status product-hunt-scraper
   sudo journalctl -u product-hunt-scraper -f
   ```

## Monitoring

### Check if cron job is running:
```bash
ps aux | grep enhanced_scraper_cron
```

### View logs:
```bash
# Python logs
tail -f scraper_cron.log

# Shell logs (if using shell script)
tail -f logs/cron_runner.log

# System logs (if using systemd)
sudo journalctl -u product-hunt-scraper -f
```

### Stop the cron job:
```bash
# If running in terminal
Ctrl+C

# If running as system service
sudo systemctl stop product-hunt-scraper
```

## Requirements

Make sure you have the required dependencies:
```bash
pip install playwright pandas playwright-stealth
playwright install chromium
```

## Troubleshooting

### Common Issues:

1. **Permission Denied**: Make shell script executable:
   ```bash
   chmod +x run_cron_scraper.sh
   ```

2. **CSV File Not Found**: Ensure `PH_check1.csv` exists in the same directory

3. **Playwright Issues**: Reinstall playwright:
   ```bash
   pip uninstall playwright
   pip install playwright
   playwright install chromium
   ```

4. **Memory Issues**: Reduce `SAMPLE_SIZE` in configuration

### Log Analysis:
- Check `scraper_cron.log` for Python errors
- Check `logs/cron_runner.log` for shell script issues
- Look for "Error in cron job" messages for retry attempts

## Customization

### Change Interval:
Edit `CRON_INTERVAL_HOURS` in the script:
```python
CRON_INTERVAL_HOURS = 6  # Run every 6 hours instead
```

### Change Sample Size:
```python
SAMPLE_SIZE = 100  # Scrape fewer URLs per run
```

### Add Email Notifications:
You can extend the script to send email notifications on completion or errors.

## Security Notes

- The script runs with the permissions of the user executing it
- Ensure the CSV file and output directory have appropriate permissions
- Consider running as a dedicated user for production deployments
- Review logs regularly for any security-related issues 