"""
AHA (Animal Harm Assessment) evaluation integration for Inspect Evals.

This module exports the key components of the AHA evaluation:
- aha_evaluation: The main task function decorated with @task.
- record_to_sample: A helper to convert raw records into evaluation samples.
- SOLVER_INSTRUCTIONS: The prompt/instructions used for scoring.
- final_digit_model_graded_qa: The scoring function for judge evaluation.
"""

from inspect_evals.aha.aha import aha_evaluation, record_to_sample
from inspect_evals.aha.scoring import SOLVER_INSTRUCTIONS, final_digit_model_graded_qa
from inspect_evals.aha.utils import setup_logging, mkd, read_json, write_json, get_latest_file, combine_csv_files


__all__ = [
    "aha_evaluation",
    "record_to_sample",
    "SOLVER_INSTRUCTIONS",
    "final_digit_model_graded_qa",
    "setup_logging",
    "mkd",
    "read_json",
    "get_latest_file",
    "combine_csv_files"
]
