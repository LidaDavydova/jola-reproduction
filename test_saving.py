import json
import sys
from datetime import datetime

from clearml import Task


def create_json_file(data, output_path):
    """Create JSON file from given data."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False, kj)
    print(f"[INFO] JSON file created successfully: {output_path}")
    return True


def main():
    try:
        # Initialize ClearML task
        task = Task.init(
            project_name="JSON Utilities",
            task_name="Create JSON File"
        )
        logger = task.get_logger()
        print("[INFO] ClearML task initialized")

    except Exception as e:
        print(f"[ERROR] Failed to initialize ClearML task: {e}")
        sys.exit(1)

    try:
        # Example data
        data = {
            "created_at": datetime.utcnow().isoformat(),
            "status": "success",
            "values": [1, 2, 3, 4, 5]
        }

        output_path = "output.json"

        # Log parameters
        task.connect({"output_path": output_path})

        success = create_json_file(data, output_path)

        task.upload_artifact(
            name="output_json",
            artifact_object="output.json"
        )

        # Log result
        logger.report_text(f"JSON creation success: {success}")
        logger.report_scalar(
            title="json_status",
            series="success",
            value=1 if success else 0,
            iteration=0
        )

        if not success:
            print("[ERROR] JSON creation failed")
            sys.exit(1)

        print("[INFO] Script finished successfully")

    except Exception as e:
        print(f"[ERROR] Unexpected error during execution: {e}")
        logger.report_text(f"Execution error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
