import json
import uuid
from datetime import datetime, timezone

from core.config import AUDIT_DIR


class AuditLogger:
    def __init__(self, run_id=None):
        self.run_id = run_id or str(uuid.uuid4())
        self.log_path = AUDIT_DIR / f"{self.run_id}.jsonl"

    def log(self, agent, action, **kwargs):
        entry = {
            **kwargs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "agent": agent,
            "action": action,
        }
        with self.log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry) + "\n")

    def get_logs(self):
        if not self.log_path.exists():
            return []

        with self.log_path.open("r", encoding="utf-8") as log_file:
            return [json.loads(line) for line in log_file if line.strip()]