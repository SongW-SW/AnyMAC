# AnyMAC


## Setup Instructions

### 1. Install and Run Ollama

Ollama needs to be installed and running to use local LLMs:

```bash
# For macOS
brew install ollama

# For Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Start the Ollama server:
```bash
ollama serve
```

Keep this terminal window open while running experiments.

### 2. Activate Conda Environment

```bash
conda activate gdesigner
```

### 3. Run Benchmarks

The benchmark script supports both OpenAI models (gpt-4o, o3-mini) and local models via Ollama.

```bash
# Run with default model (deepseek-r1:1.5b-qwen-distill-fp16)
python AnyMAC/experiments/run_llm_benchmark.py

# Run with a specific model
python AnyMAC/experiments/run_llm_benchmark.py --llm_name qwen2.5:1.5b-instruct-fp16

# Run with multiple models
python AnyMAC/experiments/run_llm_benchmark.py --models gpt-4o llama3.2:1b-instruct-fp16 gemma3:1b

# Run with all supported models
python AnyMAC/experiments/run_llm_benchmark.py --all_models
```

## Supported Models

### OpenAI Models
- gpt-4o
- o3-mini

### Local Models (Ollama)
- qwen2.5:0.5b-instruct-fp16
- llama3.2:1b-instruct-fp16
- qwen2.5:1.5b-instruct-fp16
- gemma3:1b
- deepseek-r1:1.5b-qwen-distill-fp16

## Advanced Options

```bash
# Set batch size
python AnyMAC/experiments/run_llm_benchmark.py --batch_size 8

# Set number of iterations
python AnyMAC/experiments/run_llm_benchmark.py --num_iterations 20

# Change routing mode
python AnyMAC/experiments/run_llm_benchmark.py --mode DirectAnswer
```

Available modes:
- DirectAnswer
- FullConnected
- Random
- Chain
- Debate
- Layered
- Star

## Results

Results are saved in `AnyMAC/result/gsm8k/` with filenames that include the model name and timestamp. 
