"""
HalluciGuard — run.py
Convenience launcher for backend + frontend.
"""
import subprocess
import sys
import os
from pathlib import Path


def main():
    print("================================")
    print("  HalluciGuard Launcher")
    print("================================")
    print()
    print("This launcher starts both backend and frontend.")
    print("For development, start each in separate terminals:")
    print()
    print("  Terminal 1 (Backend):")
    print("    uvicorn backend.main:app --reload --port 8000")
    print()
    print("  Terminal 2 (Frontend):")
    print("    streamlit run frontend/app.py")
    print()
    print("  Terminal 3 (MLflow UI):")
    print("    mlflow ui --port 5000")
    print()
    print("  Experiments:")
    print("    python experiments/run_experiment.py --method adaptive")
    print("    python experiments/run_experiment.py --method all")
    print()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        if cmd == "backend":
            os.execvp("uvicorn", ["uvicorn", "backend.main:app", "--reload", "--port", "8000"])
        elif cmd == "frontend":
            os.execvp("streamlit", ["streamlit", "run", "frontend/app.py"])
        elif cmd == "mlflow":
            os.execvp("mlflow", ["mlflow", "ui", "--port", "5000"])
        elif cmd == "test":
            os.execvp("pytest", ["pytest", "tests/", "-v"])
        else:
            print(f"Unknown command: {cmd}")
            print("Usage: python run.py [backend|frontend|mlflow|test]")


if __name__ == "__main__":
    main()
