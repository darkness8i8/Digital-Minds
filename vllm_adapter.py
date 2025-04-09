#!/usr/bin/env python
"""
Adapter script to run AHA with Hugging Face models using vLLM.
Sets environment variables for inspect-ai's vLLM provider.
"""
import os
import sys
import logging
import argparse
import aha # Import aha to call its main function

# Set up logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Use a temporary parser to check the --model argument
    # We use add_help=False and parse_known_args to avoid conflicts with aha.py's parser
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--model")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    is_hf_model = False
    if pre_args.model and '/' in pre_args.model:
        parts = pre_args.model.split('/')
        known_providers = {'anthropic', 'openai', 'google', 'cohere'}
        # Check if it looks like org/model and not a known provider
        if len(parts) == 2 and parts[0].lower() not in known_providers:
            is_hf_model = True

    if is_hf_model:
        logging.info(f"Setting up environment for vLLM with HuggingFace model: {pre_args.model}")

        # Set the provider BEFORE inspect_ai initializes
        os.environ["INSPECT_EVAL_PROVIDER"] = "vllm"

        # Parse specific args needed for VLLM env vars, again using parse_known_args
        vllm_arg_parser = argparse.ArgumentParser(add_help=False)
        vllm_arg_parser.add_argument("--model_temperature")
        vllm_arg_parser.add_argument("--seed")
        vllm_args, _ = vllm_arg_parser.parse_known_args()

        # Set VLLM env vars if provided
        if vllm_args.model_temperature is not None and vllm_args.model_temperature.lower() != "none":
             os.environ["VLLM_TEMPERATURE"] = vllm_args.model_temperature
        if vllm_args.seed is not None:
             os.environ["VLLM_SEED"] = str(vllm_args.seed)

        # Set other common VLLM defaults if needed
        os.environ.setdefault("VLLM_TP_SIZE", "1")
        os.environ.setdefault("VLLM_MAX_TOKENS", "1000") # Match aha.py's generate() default

    else:
        logging.info(f"Model '{pre_args.model}' doesn't look like a HuggingFace model. Using default provider resolution.")
        # Let aha.py and inspect-ai handle provider resolution based on model name or --openai-base-url

    # Important: Set sys.argv correctly for aha.py's argparse
    # sys.argv[0] should be the script name, followed by all original arguments
    original_argv = sys.argv[1:]
    sys.argv = [f"{os.path.dirname(__file__)}/aha.py"] + original_argv

    # Call the main function from aha.py
    try:
        aha.main()
    except Exception as e:
        logging.error(f"Error during aha.main(): {e}")
        # Re-raise the exception after logging if needed
        raise
