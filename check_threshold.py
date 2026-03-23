import mlflow
import sys
import os

THRESHOLD = 0.99

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

print(f"Checking Run ID: {run_id}")

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)
accuracy = run.data.metrics.get("accuracy", 0.0)

print(f"Accuracy: {accuracy}")
print(f"Threshold: {THRESHOLD}")

if accuracy < THRESHOLD:
    print(f"FAILED: accuracy {accuracy:.4f} is below threshold {THRESHOLD}")
    sys.exit(1)

print(f"PASSED: accuracy {accuracy:.4f} meets threshold {THRESHOLD}")
