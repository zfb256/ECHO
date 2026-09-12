"""Offline API wiring and resume check with synthetic inputs."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ApiMockTest(unittest.TestCase):
    def test_mock_and_resume(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            directory = Path(directory)
            tasks = directory / "tasks.jsonl"
            output = directory / "outputs_mock.jsonl"
            tasks.write_text("".join(json.dumps({
                "pair_id": "synthetic", "condition": condition,
                "prompt": "Synthetic test input", "model": "local-test",
            }) + "\n" for condition in ("clean", "contaminated", "clean_b")))
            command = [sys.executable, "-B", str(ROOT / "scripts/run_api_inference.py"),
                       "--tasks", str(tasks), "--model", "deepseek-v4-flash",
                       "--output", str(output), "--mock"]
            subprocess.run(command, cwd=directory, check=True, capture_output=True)
            before = output.read_bytes()
            rows = [json.loads(line) for line in before.splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row["response"].startswith("[MOCK") for row in rows))
            subprocess.run(command, cwd=directory, check=True, capture_output=True)
            self.assertEqual(before, output.read_bytes())


if __name__ == "__main__":
    unittest.main()
