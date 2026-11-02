"""CPU tests for scripts/runpod/fleet.py with a fake runner: no RunPod calls."""
import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fleet", ROOT / "scripts/runpod/fleet.py")
fleet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fleet)

GPU_LIST = json.dumps([
    {"gpuId": "NVIDIA GeForce RTX 4090", "communityCloud": False, "secureCloud": True, "available": True,
     "communityPricePerHr": None, "securePricePerHr": 0.89},
    {"gpuId": "NVIDIA GeForce RTX 3090", "communityCloud": True, "secureCloud": True, "available": True,
     "communityPricePerHr": 0.22, "securePricePerHr": 0.5},
    {"gpuId": "NVIDIA L4", "communityCloud": True, "secureCloud": True, "available": False},
])


class FakeRunner(fleet.Runner):
    """Answers runpodctl/ssh/rsync like RunPod would; records every command."""

    def __init__(self, local_status: Path | None = None, out_of_stock=(), fail_outputs=False):
        super().__init__(dry_run=False)
        self.fail_outputs = fail_outputs
        self.commands = []
        self.local_status = local_status
        self.out_of_stock = set(out_of_stock)
        self.created = 0

    def __call__(self, command, *, check=True, input_text=None):
        self.commands.append(command)
        head = command[:3]
        if head == ["runpodctl", "gpu", "list"]:
            return GPU_LIST
        if head == ["runpodctl", "pod", "create"]:
            gpu = command[command.index("--gpu-id") + 1]
            cloud = command[command.index("--cloud-type") + 1]
            if (gpu, cloud) in self.out_of_stock:
                raise RuntimeError("There are no longer any instances available")
            self.created += 1
            return json.dumps({"id": f"pod{self.created}", "costPerHr": 0.34})
        if head == ["runpodctl", "ssh", "info"]:  # v3.0.1 lags behind `pod get`
            return json.dumps({"error": "pod not ready: no container reported yet"})
        if head == ["runpodctl", "pod", "get"]:
            return json.dumps({"id": command[3], "costPerHr": 0.34, "gpuId": "NVIDIA GeForce RTX 4090",
                               "ssh": {"id": command[3], "ip": "203.0.113.7", "port": 10022,
                                       "ssh_command": "ssh -i k root@203.0.113.7 -p 10022"}})
        if command[0] == "ssh" and command[-1] == "echo ready":
            return "ready\n"
        if command[0] == "rsync" and "/results/" in command[-2] and self.fail_outputs:
            raise RuntimeError("rsync failed (12): connection unexpectedly closed")
        if command[0] == "rsync" and command[-2].endswith(":/workspace/status/") and self.local_status:
            Path(command[-1]).mkdir(parents=True, exist_ok=True)
            (Path(command[-1]) / "DONE").write_text("2026-11-01T00:00:00Z\n")
        return ""


class FleetTests(unittest.TestCase):
    def test_ssh_info_parsing_ignores_the_proxy(self):
        self.assertEqual(fleet.parse_ssh_info("ssh root@38.80.152.73 -p 32061 -i ~/.ssh/id_ed25519"),
                         {"user": "root", "host": "38.80.152.73", "port": 32061})
        self.assertEqual(fleet.parse_ssh_info(json.dumps({"ip": "38.80.152.73", "port": "32061"})),
                         {"user": "root", "host": "38.80.152.73", "port": 32061})
        self.assertIsNone(fleet.parse_ssh_info("ssh 8y5rumuyb50m78-6441103b@ssh.runpod.io -i key"))
        self.assertIsNone(fleet.parse_ssh_info(""))

    def test_candidates_are_cheapest_first_and_skip_unavailable(self):
        gpus = ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 3090", "NVIDIA L4", "NVIDIA A40"]
        pairs = fleet.candidates(gpus, ["COMMUNITY", "SECURE"], GPU_LIST)
        self.assertEqual(pairs[:3], [("NVIDIA GeForce RTX 3090", "COMMUNITY", 0.22),
                                     ("NVIDIA GeForce RTX 3090", "SECURE", 0.5),
                                     ("NVIDIA GeForce RTX 4090", "SECURE", 0.89)])
        self.assertNotIn("NVIDIA L4", {gpu for gpu, _cloud, _price in pairs})
        self.assertEqual(pairs[3:], [("NVIDIA A40", "COMMUNITY", None), ("NVIDIA A40", "SECURE", None)])
        capped = fleet.candidates(gpus, ["COMMUNITY", "SECURE"], GPU_LIST, max_price=0.6)
        self.assertNotIn(("NVIDIA GeForce RTX 4090", "SECURE", 0.89), capped)
        # Unknown listing output keeps every pair rather than guessing.
        self.assertEqual(len(fleet.candidates(["X"], ["COMMUNITY", "SECURE"], "not json")), 2)

    def test_shard_job_is_valid_shell_and_runs_the_right_shard(self):
        state = {"grid": "grids/calibration.toml", "shards": 3, "max_batch": 64, "model_batch_size": 32}
        job = fleet.shard_job(state, 2)
        subprocess.run(["bash", "-n", "-c", job], check=True)
        self.assertIn("run_shard.sh grids/calibration.toml 2/3", job)
        self.assertIn("bootstrap.sh", job)
        self.assertTrue(job.rstrip().endswith("& }"))
        # Run it against a fake workspace: the log must appear even though logs/ did not exist.
        with tempfile.TemporaryDirectory() as directory:
            fake = job.replace(fleet.REMOTE_WORK, directory).replace(
                f"cd {directory}/structure-asr", f"cd {directory}").replace(
                "bash scripts/runpod/bootstrap.sh", "true").replace(
                "bash scripts/runpod/run_shard.sh", "echo run_shard")
            # The ssh session must not wait for the job: the command returns at once.
            subprocess.run(["bash", "-c", fake.replace("echo run_shard", "sleep 3; echo run_shard")],
                           check=True, timeout=2, stdout=subprocess.PIPE)
            subprocess.run(["sleep", "3.5"])
            self.assertIn("run_shard grids/calibration.toml 2/3",
                          (Path(directory) / "logs" / "shard.log").read_text())

    def test_launch_falls_back_and_sync_terminates_finished_pods(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(fleet, "run_dir", lambda run: Path(directory) / run), \
                patch.object(fleet, "committed_head", lambda grid: "abc123"):
            runner = FakeRunner(local_status=Path(directory),
                                out_of_stock={("NVIDIA GeForce RTX 3090", "COMMUNITY")})
            fleet.main(["--ssh-key", "/k", "launch", "cal", "--grid", "grids/calibration.toml",
                        "--pods", "2", "--yes"], runner=runner)
            state = json.loads((Path(directory) / "cal" / "fleet.json").read_text())
            self.assertEqual(state["commit"], "abc123")
            self.assertEqual(state["outdir"], "results/grids/calibration")
            self.assertEqual({(pod["gpu"], pod["cloud"]) for pod in state["pods"].values()},
                             {("NVIDIA GeForce RTX 3090", "SECURE")})
            self.assertEqual(state["pods"]["1"]["ssh"]["port"], 10022)
            remote = [c[-1] for c in runner.commands if c[0] == "ssh"]
            self.assertTrue(any("git checkout -q abc123" in r for r in remote))
            self.assertTrue(any("run_shard.sh grids/calibration.toml 1/2" in r for r in remote))
            self.assertTrue(all("-i" in c for c in runner.commands if c[0] == "ssh"))

            runner.fail_outputs = True  # DONE synced but outputs did not: keep the pods.
            fleet.main(["--ssh-key", "/k", "sync", "cal", "--terminate-done"], runner=runner)
            self.assertFalse([c for c in runner.commands if c[:3] == ["runpodctl", "pod", "delete"]])
            runner.fail_outputs = False
            fleet.main(["--ssh-key", "/k", "sync", "cal", "--terminate-done"], runner=runner)
            deleted = [c[3] for c in runner.commands if c[:3] == ["runpodctl", "pod", "delete"]]
            self.assertEqual(sorted(deleted), ["pod1", "pod2"])
            state = json.loads((Path(directory) / "cal" / "fleet.json").read_text())
            self.assertTrue(all(pod["terminated"] == "done" for pod in state["pods"].values()))
            # relaunch on a run with nothing synced yet must not rsync a
            # missing directory (it used to strand the new pod).
            fleet.main(["--ssh-key", "/k", "relaunch", "cal", "--shard", "0"], runner=runner)
            pushes = [c for c in runner.commands
                      if c[0] == "rsync" and c[-2].rstrip("/").endswith("grid")]
            self.assertEqual(pushes, [])
            self.assertFalse(json.loads((Path(directory) / "cal" / "fleet.json").read_text())
                             ["pods"]["0"].get("terminated"))
            # An existing pod can be adopted instead of created.
            before = runner.created
            fleet.main(["--ssh-key", "/k", "launch", "cal2", "--grid", "grids/calibration.toml",
                        "--pods", "1", "--attach", "existingpod", "--yes"], runner=runner)
            self.assertEqual(runner.created, before)
            adopted = json.loads((Path(directory) / "cal2" / "fleet.json").read_text())["pods"]["0"]
            self.assertEqual((adopted["id"], adopted["ssh"]["port"]), ("existingpod", 10022))
            with self.assertRaises(SystemExit):  # A run name is never reused.
                fleet.main(["launch", "cal", "--grid", "grids/calibration.toml", "--yes"], runner=runner)


if __name__ == "__main__":
    unittest.main()
