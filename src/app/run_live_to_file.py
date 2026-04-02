import subprocess
from datetime import datetime

# file output
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"logs/live_{timestamp}.txt"

# esegui script live
result = subprocess.run(
    ["python", "-m", "src.app.run_live"],
    capture_output=True,
    text=True
)

# salva output
with open(log_file, "w") as f:
    f.write(result.stdout)

print(f"Saved to {log_file}")