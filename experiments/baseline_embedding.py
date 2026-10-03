"""Embedding baseline experiment."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.utils.logging_config import setup_logging
setup_logging()

from evaluation.evaluator import run_evaluation

if __name__ == "__main__":
    result = run_evaluation(method="embedding")
    print("Embedding baseline complete:", result["classification"])
