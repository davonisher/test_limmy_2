# 🚀 GPU Optimization Guide for High-Speed Affiliation Extraction

## ⚡ Ollama Configuration for Maximum Speed

### 1. GPU Memory Settings
Configure Ollama to use your GPU optimally:

```bash
# Set GPU layers (adjust based on your GPU memory)
export OLLAMA_NUM_GPU_LAYERS=35  # For most modern GPUs
# OR for maximum GPU usage:
export OLLAMA_NUM_GPU_LAYERS=-1  # Use all available layers on GPU

# Increase context size and batch size
export OLLAMA_NUM_CTX=4096       # Context window
export OLLAMA_NUM_BATCH=512      # Batch size for processing
export OLLAMA_NUM_PREDICT=1024   # Max prediction tokens
```

### 2. Model Selection
For fastest processing, consider these model options:

```bash
# Fastest (recommended for high-throughput):
ollama pull llama3.2:3b         # Smallest, fastest
ollama pull llama3.3:latest     # Good balance of speed/quality

# If you need higher quality and have strong GPU:
ollama pull llama3.2:8b         # Larger but still fast
```

### 3. System Optimization

#### GPU Memory
```bash
# Check GPU memory usage
nvidia-smi

# For maximum performance, ensure:
# - GPU has at least 8GB VRAM for llama3.3
# - Close other GPU-intensive applications
# - Use GPU-optimized drivers
```

#### System Resources
```bash
# Increase file descriptors for high concurrency
ulimit -n 65536

# Monitor system resources
htop
nvtop  # For GPU monitoring
```

## 🏃‍♂️ Running the Optimized Extractor

### Quick Performance Test
```bash
cd scientist/
python test_extraction_performance.py
```

### Full Dataset Processing
```bash
cd scientist/
python extract_companies_universities.py
```

## 📊 Expected Performance Metrics

| GPU Type | Scientists/Second | Time for 10k Scientists |
|----------|------------------|------------------------|
| RTX 4090 | 8-12            | 15-20 minutes         |
| RTX 4080 | 6-10            | 20-25 minutes         |
| RTX 3090 | 5-8             | 25-35 minutes         |
| RTX 3080 | 4-6             | 30-45 minutes         |

## 🔧 Troubleshooting

### If Processing is Still Slow:

1. **Check Ollama is using GPU:**
   ```bash
   ollama ps  # Should show GPU usage
   nvidia-smi  # Check GPU utilization
   ```

2. **Increase concurrency** (if GPU can handle it):
   ```python
   # In extract_companies_universities.py, increase:
   max_concurrent = 200  # From 150 to 200
   batch_size = 150      # From 100 to 150
   ```

3. **Reduce model context:**
   ```python
   # In the options dict:
   "num_ctx": 1024  # Reduce from 2048
   ```

4. **Use smaller model:**
   ```python
   # Change model to:
   extractor = ScientistAffiliationExtractor(csv_path, model="llama3.2:3b")
   ```

## 🎯 Performance Monitoring

Monitor your extraction with:
```bash
# In another terminal:
watch -n 1 'nvidia-smi; echo "---"; ps aux | grep ollama'
```

## 💡 Pro Tips

1. **Warm up Ollama** before processing:
   ```bash
   ollama run llama3.3:latest "Test prompt"
   ```

2. **Process during off-peak hours** for maximum GPU availability

3. **Use SSD storage** for faster intermediate saves

4. **Close browser/other GPU apps** during processing

5. **Consider using multiple models** in parallel if you have multiple GPUs

## 📈 Scaling Further

For even faster processing with multiple GPUs or machines:

1. **Multi-GPU setup:** Run multiple Ollama instances
2. **Distributed processing:** Split dataset across machines
3. **Batch processing:** Process in chunks and merge results

---

🚀 **Happy high-speed extraction!** Your strong GPU should now be fully utilized for maximum throughput. 