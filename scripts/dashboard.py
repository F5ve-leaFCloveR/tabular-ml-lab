import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))


if __name__ == "__main__":
    port = os.getenv("STREAMLIT_PORT", "8501")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{env.get('PYTHONPATH', '')}"
    subprocess.run(["streamlit", "run", "app/dashboard.py", "--server.port", port], env=env)
