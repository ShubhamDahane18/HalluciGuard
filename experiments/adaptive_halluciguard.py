"""Adaptive HalluciGuard experiment — proposed method."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.utils.logging_config import setup_logging
setup_logging()

from evaluation.evaluator import run_evaluation

if __name__ == "__main__":
    result = run_evaluation(method="adaptive")
    print("Adaptive HalluciGuard complete:", result["classification"])
    eff = result["efficiency"]
    print(f"LLM avoidance rate: {eff['llm_avoidance_rate']:.1%}")
    print(f"LLM calls: {eff['llm_calls']} / {eff['total_claims']}")
