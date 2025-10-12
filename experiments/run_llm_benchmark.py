import argparse
import subprocess
import os
import sys
from typing import List

# Models supported
OPENAI_MODELS = ["gpt-4o-2024-11-20", "o3-mini", "gpt-4-1106-preview"]
SMALL_LLM_MODELS = [
    "qwen2.5:0.5b-instruct-fp16",
    "llama3.2:1b-instruct-fp16",
    "qwen2.5:1.5b-instruct-fp16",
    # "gemma3:1b-it-fp16", not working
    "qwen3:0.6b-fp16",
    "temp0-qwen2.5-1.5b",
    "deepseek-r1:1.5b-qwen-distill-fp16"  # Default model
]

def run_gsm8k_benchmark(llm_name: str, mode: str = "FullConnected", method: str = "NAP"):
    """
    Run the GSM8K benchmark with the specified LLM model
    """
    cmd = [
        "python", f"{method}/experiments/run_gsm8k.py",
        "--llm_name", llm_name,
        "--mode", mode
    ]
    
    # Set environment variables based on model type
    env = os.environ.copy()
    if llm_name in OPENAI_MODELS:
        env["BASE_URL"] = "https://api.openai.com/v1"
    else:
        env["BASE_URL"] = "http://localhost:11434"
        env["API_KEY"] = ""
    
    print(f"Running benchmark with model: {llm_name}")
    subprocess.run(cmd, env=env)

def run_mmlu_benchmark(llm_name: str, mode: str = "FullConnected", method: str = "NAP"):
    """
    Run the MMLU benchmark with the specified LLM model
    """
    cmd = [
        "python", f"{method}/experiments/run_mmlu.py",
        "--llm_name", llm_name,
        "--mode", mode
    ]
    
    # Set environment variables based on model type
    env = os.environ.copy()
    if llm_name in OPENAI_MODELS:
        env["BASE_URL"] = "https://api.openai.com/v1"
    else:
        env["BASE_URL"] = "http://localhost:11434"
        env["API_KEY"] = ""
        
    subprocess.run(cmd, env=env)

def run_humaneval_benchmark(llm_name: str, mode: str = "FullConnected", method: str = "NAP"):
    """
    Run the HumanEval benchmark with the specified LLM model
    """
    cmd = [
        "python", f"{method}/experiments/run_humaneval.py",
        "--llm_name", llm_name,
        "--mode", mode
    ]
    env = os.environ.copy()
    
    if llm_name in OPENAI_MODELS:
        env["BASE_URL"] = "https://api.openai.com/v1"
    else:
        env["BASE_URL"] = "http://localhost:11434"
        env["API_KEY"] = ""
        
    subprocess.run(cmd, env=env)

def run_multiarith_benchmark(llm_name: str, mode: str = "FullConnected", method: str = "NAP"):
    """
    Run the HumanEval benchmark with the specified LLM model
    """
    cmd = [
        "python", f"{method}/experiments/run_multiarith.py",
        "--llm_name", llm_name,
        "--mode", mode
    ]
    env = os.environ.copy()
    
    if llm_name in OPENAI_MODELS:
        env["BASE_URL"] = "https://api.openai.com/v1"
    else:
        env["BASE_URL"] = "http://localhost:11434"
        env["API_KEY"] = ""
        
    subprocess.run(cmd, env=env)

def run_svamp_benchmark(llm_name: str, mode: str = "FullConnected", method: str = "NAP"):
    """
    Run the SVAMP benchmark with the specified LLM model
    """
    cmd = [
        "python", f"{method}/experiments/run_svamp.py",
        "--llm_name", llm_name,
        "--mode", mode
    ]
    env = os.environ.copy()
    
    if llm_name in OPENAI_MODELS:
        env["BASE_URL"] = "https://api.openai.com/v1"
    else:
        env["BASE_URL"] = "http://localhost:11434"
        env["API_KEY"] = ""
        
    subprocess.run(cmd, env=env)

def run_aqua_rat_benchmark(llm_name: str, mode: str = "FullConnected", method: str = "NAP"):    
    """
    Run the AQUA_RAT benchmark with the specified LLM model
    """
    cmd = [
        "python", f"{method}/experiments/run_aqua_rat.py",
        "--llm_name", llm_name,
        "--mode", mode
    ]
    env = os.environ.copy()
    
    if llm_name in OPENAI_MODELS:
        env["BASE_URL"] = "https://api.openai.com/v1"
    else:
        env["BASE_URL"] = "http://localhost:11434"
        env["API_KEY"] = ""
        
    subprocess.run(cmd, env=env)

def run_mmlu_liar_benchmark(llm_name: str, mode: str = "FullConnected", method: str = "NAP"):
    """
    Run the MMLU_Liar benchmark with the specified LLM model
    """
    cmd = [
        "python", f"{method}/experiments/run_mmlu_liar.py",
        "--llm_name", llm_name,
        "--mode", mode
    ]
    env = os.environ.copy()
    
    if llm_name in OPENAI_MODELS:
        env["BASE_URL"] = "https://api.openai.com/v1"
    else:
        env["BASE_URL"] = "http://localhost:11434"
        env["API_KEY"] = ""
        
    subprocess.run(cmd, env=env)

def run_mmlu_eff_benchmark(llm_name: str, mode: str = "FullConnected", method: str = "NAP"):
    """
    Run the MMLU_EFF benchmark with the specified LLM model
    """
    cmd = [
        "python", f"{method}/experiments/run_mmlu_eff.py",
        "--llm_name", llm_name,
        "--mode", mode
    ]
    env = os.environ.copy()
    
    if llm_name in OPENAI_MODELS:
        env["BASE_URL"] = "https://api.openai.com/v1"
    else:
        env["BASE_URL"] = "http://localhost:11434"
        env["API_KEY"] = ""
    
    subprocess.run(cmd, env=env)

def run_gsm8k_eff_benchmark(llm_name: str, mode: str = "FullConnected", method: str = "NAP"):
    """
    Run the GSM8K_EFF benchmark with the specified LLM model
    """
    cmd = [
        "python", f"{method}/experiments/run_gsm8k_eff.py",
        "--llm_name", llm_name,
        "--mode", mode
    ]
    env = os.environ.copy()
    
    if llm_name in OPENAI_MODELS:
        env["BASE_URL"] = "https://api.openai.com/v1"
    else:
        env["BASE_URL"] = "http://localhost:11434"
        env["API_KEY"] = ""
        
    subprocess.run(cmd, env=env)

def run_svamp_eff_benchmark(llm_name: str, mode: str = "FullConnected", method: str = "NAP"):
    """
    Run the SVAMP_EFF benchmark with the specified LLM model
    """
    cmd = [
        "python", f"{method}/experiments/run_svamp_eff.py",
        "--llm_name", llm_name,
        "--mode", mode
    ]
    env = os.environ.copy()
    
    if llm_name in OPENAI_MODELS:
        env["BASE_URL"] = "https://api.openai.com/v1"
    else:
        env["BASE_URL"] = "http://localhost:11434"
        env["API_KEY"] = ""
        
    subprocess.run(cmd, env=env)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GSM8K benchmarks with different LLM models")
    parser.add_argument("--llm_name", type=str, default="deepseek-r1:1.5b-qwen-distill-fp16",
                        help="Name of the LLM model to use")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=['gsm8k', 'mmlu', 'humaneval', 'multiarith', 'svamp', 'aqua_rat', 'mmlu_liar', 'mmlu_eff', 'gsm8k_eff', 'svamp_eff'],
                        help="Dataset to use for benchmark")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for processing")
    parser.add_argument("--num_iterations", type=int, default=10,
                        help="Number of iterations for benchmark")
    parser.add_argument("--mode", type=str, default="FullConnected",
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star'],
                        help="Mode of operation")
    parser.add_argument("--models", nargs="+", type=str,
                        help="Specific models to run benchmarks with")
    parser.add_argument("--method", type=str, default="GDesigner",
                        choices=['GDesigner', 'NAP'],
                        help="Method to use for benchmark")
    
    args = parser.parse_args()
    
    if args.dataset == "gsm8k":
        run_gsm8k_benchmark(args.llm_name, args.mode, args.method)
    elif args.dataset == "mmlu":
        run_mmlu_benchmark(args.llm_name, args.mode, args.method)
    elif args.dataset == "humaneval":
        run_humaneval_benchmark(args.llm_name, args.mode, args.method)
    elif args.dataset == "multiarith":
        run_multiarith_benchmark(args.llm_name, args.mode, args.method)
    elif args.dataset == "svamp":
        run_svamp_benchmark(args.llm_name, args.mode, args.method)
    elif args.dataset == "aqua_rat":
        run_aqua_rat_benchmark(args.llm_name, args.mode, args.method)
    elif args.dataset == "mmlu_liar":
        run_mmlu_liar_benchmark(args.llm_name, args.mode, args.method)
    elif args.dataset == "mmlu_eff":
        run_mmlu_eff_benchmark(args.llm_name, args.mode, args.method)
    elif args.dataset == "gsm8k_eff":
        run_gsm8k_eff_benchmark(args.llm_name, args.mode, args.method)
    elif args.dataset == "svamp_eff":
        run_svamp_eff_benchmark(args.llm_name, args.mode, args.method)
