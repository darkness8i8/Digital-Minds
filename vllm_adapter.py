"""
Adapter script to run AHA with Hugging Face models using vLLM.
"""
import os
import sys
import json
import logging
import importlib
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)

def run_with_vllm(args):
    """
    Run AHA with a Hugging Face model using vLLM directly.
    """
    # First, set environment variable to use vLLM provider
    os.environ["INSPECT_EVAL_PROVIDER"] = "vllm"
    
    # Set the model to use
    model_name = args.model
    os.environ["INSPECT_EVAL_MODEL"] = model_name
    
    # Set vLLM-specific parameters
    if args.model_temperature is not None and args.model_temperature.lower() != "none":
        os.environ["VLLM_TEMPERATURE"] = args.model_temperature
    
    # Set additional vLLM parameters
    if args.seed is not None:
        os.environ["VLLM_SEED"] = str(args.seed)
    
    # Default to 1 for TP degree if not set
    if not os.environ.get("VLLM_TP_SIZE"):
        os.environ["VLLM_TP_SIZE"] = "1"
    
    # Add max tokens setting
    os.environ["VLLM_MAX_TOKENS"] = "1024"
    
    # Pass through to main script
    from aha import main
    main()

if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser("AHA Benchmark with vLLM")
    parser.add_argument("--model", required=True, help="Hugging Face model ID (e.g., meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--judges", default="anthropic/claude-3-5-sonnet-20241022,google/gemini-1.5-pro-002,openai/gpt-4o-2024-08-06")
    parser.add_argument("--dataset")
    parser.add_argument("--output_dir")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--model_temperature", default="0.5")
    parser.add_argument("--judge_temperature", default="0.5")
    parser.add_argument("--run-analysis", action="store_true")
    parser.add_argument("--start-batch", type=int, default=0)
    parser.add_argument("--calculate_judge_harshness", action="store_true", help="If set, judge harshness is calculated from data.")
    
    args = parser.parse_args()
    
    # Check if model is a HF model
    if '/' in args.model:
        parts = args.model.split('/')
        known_providers = {'anthropic', 'openai', 'google', 'cohere'}
        if len(parts) == 2 and parts[0].lower() not in known_providers:
            logging.info(f"Using vLLM for HuggingFace model: {args.model}")
            run_with_vllm(args)
        else:
            logging.info(f"Using standard provider for model: {args.model}")
            from aha import main
            main()
    else:
        logging.info(f"Using standard provider for model: {args.model}")
        from aha import main
        main()
