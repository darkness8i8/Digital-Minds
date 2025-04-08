#!/usr/bin/env python
"""
Wrapper script to run AHA with HuggingFace models by monkey-patching inspect-ai.
"""
import os
import sys
import logging
import importlib

# Set up logging
logging.basicConfig(level=logging.INFO)

# Try to import inspect_ai and patch it
try:
    from inspect_ai.model import _model
    
    # Store the original get_model function
    original_get_model = _model.get_model
    
    # Define our patched version
    def patched_get_model(model, *args, **kwargs):
        if '/' in model:
            # Check if it's a HuggingFace model
            parts = model.split('/')
            known_providers = {'anthropic', 'openai', 'google', 'cohere'}
            if len(parts) == 2 and parts[0].lower() not in known_providers:
                logging.info(f"Using vLLM for HuggingFace model: {model}")
                # Use vLLM provider with HF model
                from inspect_ai.model import VLLMProvider
                return VLLMProvider(model)
        
        # Fall back to original implementation for other models
        return original_get_model(model, *args, **kwargs)
    
    # Replace the original function with our patched version
    _model.get_model = patched_get_model
    logging.info("Successfully patched inspect-ai to support HuggingFace models")
    
except ImportError:
    logging.error("Failed to patch inspect-ai. Make sure it's installed.")
    sys.exit(1)

# Import and run aha.py
if __name__ == "__main__":
    # Import the main module from aha.py
    import aha
    aha.main()
