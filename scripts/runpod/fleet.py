#!/usr/bin/env python3
"""Run one asr-grid on several RunPod GPU pods, one shard per pod.

Each pod is self-contained (no network volume): it receives a git bundle of
the exact commit, bootstraps (venv, model weights, IQ-TREE) and runs every
inference stage for its shard (scripts/runpod/run_shard.sh). Results come back
over rsync into results/runpod/<run>/, where `evaluate` runs locally.

  fleet.py launch RUN --grid grids/calibration.toml --pods 1 [--dry-run]
  fleet.py status RUN
  fleet.py sync RUN [--every 10 --terminate-done --max-hours 6]
  fleet.py logs RUN --shard 0
  fleet.py relaunch RUN --shard 0      # replace a lost pod; resumes from synced output
  fleet.py terminate RUN [--shard 0]

Requires a configured `runpodctl` and an SSH key registered with RunPod. Every
pod is billed until terminated: check `status` and terminate when done.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REMOTE_WORK = "/workspace"
REMOTE_REPO = f"{REMOTE_WORK}/structure-asr"
# Official "Runpod Pytorch 2.4.0" template: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# with its start script (sshd, PUBLIC_KEY). Creating from the bare image instead
# left containers unbooted for >10 min; from the template SSH was up in ~40 s.
DEFAULT_TEMPLATE = "runpod-torch-v240"
# CUDA 12.4 / torch 2.4 image: pods need a driver for >= 12.4, and Blackwell
# GPUs (RTX 50xx, RTX PRO Blackwell) are unsupported by torch 2.4.
MIN_CUDA = "12.4"
# Cheap pre-Blackwell types with >= 24 GB (ProstT5 fp16 needs ~6 GB). Tried
# cheapest first using `runpodctl gpu list` prices; this order breaks ties.
DEFAULT_GPUS = [
    "NVIDIA GeForce RTX 3090",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A5000",
    "NVIDIA L4",
    "NVIDIA A40",
    "NVIDIA RTX A6000",
    "NVIDIA L40S",
]
SSH_OPTIONS = [
    "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
]


class Runner:
    """All external commands go through here so tests can substitute a fake."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

    def __call__(self, command: list[str], *, check: bool = True, input_text: str | None = None) -> str:
        if self.dry_run:
            print("DRY-RUN:", shlex.join(command))
            return ""
        result = subprocess.run(command, capture_output=True, text=True, input=input_text)
        if check and result.returncode != 0:
            raise RuntimeError(f"{shlex.join(command)} failed ({result.returncode}): "
                               f"{result.stderr.strip() or result.stdout.strip()}")
        return result.stdout


# ------------------------------------------------------------------ parsing


def parse_json(text: str):
    text = text.strip()
    return json.loads(text) if text else None


def parse_ssh_info(text: str) -> dict | None:
    """Direct-TCP SSH endpoint from `runpodctl ssh info` output, or None.

    Accepts JSON (ip/publicIp + port fields) or text containing an
    `ssh root@<ip> -p <port>` command. The ssh.runpod.io proxy is ignored:
    it cannot carry scp/rsync.
    """
    try:
        payload = parse_json(text)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        host = payload.get("ip") or payload.get("publicIp") or payload.get("host")
        port = payload.get("port") or payload.get("publicPort")
        if host and port and "runpod.io" not in str(host):
            return {"user": payload.get("user", "root"), "host": str(host), "port": int(port)}
        command = payload.get("command") or payload.get("sshCommand") or ""
        text = str(command)
    match = re.search(r"ssh\s+(?:-\S+\s+\S+\s+)*(\w+)@(\d{1,3}(?:\.\d{1,3}){3})\s+-p\s+(\d+)", text)
    if match:
        return {"user": match.group(1), "host": match.group(2), "port": int(match.group(3))}
    return None


def candidates(gpu_types: list[str], clouds: list[str], gpu_list_text: str,
               max_price: float | None = None) -> list[tuple[str, str, float | None]]:
    """(gpu, cloud, $/h) to try, cheapest first.

    Pairs `runpodctl gpu list` reports unavailable or above ``max_price`` are
    dropped; unlisted pairs are kept (price unknown) after the priced ones.
    """
    try:
        listing = parse_json(gpu_list_text) or []
    except json.JSONDecodeError:
        listing = []
    known = {item.get("gpuId"): item for item in listing if isinstance(item, dict)}
    pairs = []
    for rank, gpu in enumerate(gpu_types):
        for cloud in clouds:
            info = known.get(gpu)
            price = None
            if info is not None:
                flag = "communityCloud" if cloud == "COMMUNITY" else "secureCloud"
                if info.get("available") is False or info.get(flag) is False:
                    continue
                price = info.get("communityPricePerHr" if cloud == "COMMUNITY" else "securePricePerHr")
                if max_price is not None and price is not None and price > max_price:
                    continue
            pairs.append((rank, gpu, cloud, price))
    pairs.sort(key=lambda item: (item[3] is None, item[3] or 0.0, item[0]))
    return [(gpu, cloud, price) for _rank, gpu, cloud, price in pairs]


# -------------------------------------------------------------------- state


def run_dir(run: str) -> Path:
    return ROOT / "results" / "runpod" / run


def load_state(run: str) -> dict:
    path = run_dir(run) / "fleet.json"
    if not path.exists():
        raise SystemExit(f"no fleet state at {path}")
    return json.loads(path.read_text())


def save_state(state: dict) -> None:
    path = run_dir(state["run"]) / "fleet.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def grid_outdir(grid: str) -> str:
    """The grid's output directory relative to the checkout (same on pod and here)."""
    sys.path.insert(0, str(ROOT / "src"))
    from structure_asr.grid import load_grid

    return str(load_grid(ROOT / grid).outdir.relative_to(ROOT))


# ---------------------------------------------------------------------- ssh


def ssh_command(pod: dict, remote: str, key: str | None) -> list[str]:
    ssh = pod["ssh"]
    command = ["ssh", *SSH_OPTIONS, "-p", str(ssh["port"])]
    if key:
        command += ["-i", key]
    return command + [f"{ssh['user']}@{ssh['host']}", remote]


def rsync_command(pod: dict, source: str, dest: str, key: str | None) -> list[str]:
    ssh = ["ssh", *SSH_OPTIONS, "-p", str(pod["ssh"]["port"])] + (["-i", key] if key else [])
    return ["rsync", "-rltz", "--partial", "-e", shlex.join(ssh), source, dest]


def remote(pod: dict) -> str:
    return f"{pod['ssh']['user']}@{pod['ssh']['host']}"


def shard_job(state: dict, shard: int) -> str:
    """Remote shell command: bootstrap, then run the shard, detached under nohup."""
    run_shard = shlex.join([
        "bash", "scripts/runpod/run_shard.sh", state["grid"], f"{shard}/{state['shards']}",
        "--max-batch", str(state["max_batch"]), "--model-batch-size", str(state["model_batch_size"]),
    ])
    job = (
        f"mkdir -p {REMOTE_WORK}/status {REMOTE_WORK}/logs && "
        f"(bash scripts/runpod/bootstrap.sh || "
        f"{{ echo bootstrap exit $? > {REMOTE_WORK}/status/FAILED; exit 1; }}) && {run_shard}"
    )
    # The log directory must exist before the redirect (a failed redirect on a
    # backgrounded command fails silently), and only nohup may be backgrounded:
    # a backgrounded `a && b` subshell holds the ssh session open for the whole job.
    return (f"mkdir -p {REMOTE_WORK}/logs {REMOTE_WORK}/status && cd {REMOTE_REPO} && "
            f"{{ nohup bash -c {shlex.quote(job)} > {REMOTE_WORK}/logs/shard.log 2>&1 < /dev/null & }}")


# ------------------------------------------------------------------- launch


def create_pod(runner: Runner, args, state: dict, shard: int, gpu: str, cloud: str) -> str:
    command = [
        "runpodctl", "pod", "create",
        "--name", f"asr-{state['run']}-s{shard}",
        "--template-id", args.template,
        "--gpu-id", gpu, "--gpu-count", "1",
        "--cloud-type", cloud,
        "--container-disk-in-gb", "30",
        "--volume-in-gb", str(args.volume_gb),
        "--min-cuda-version", MIN_CUDA,
        "--output", "json",
    ]
    output = runner(command)
    if runner.dry_run:
        return f"DRYRUN{shard}"
    payload = parse_json(output)
    if isinstance(payload, list):
        payload = payload[0]
    return str(payload["id"])


def pod_ssh(runner: Runner, pod_id: str) -> dict | None:
    """Direct SSH endpoint from `runpodctl pod get` (its "ssh" field).

    `runpodctl ssh info` (v3.0.1) kept answering "no container reported yet"
    for pods that `pod get` showed running with a mapped port, so it is only a
    fallback.
    """
    try:
        details = parse_json(runner(["runpodctl", "pod", "get", pod_id, "--output", "json"], check=False))
    except json.JSONDecodeError:
        details = None
    if isinstance(details, dict) and isinstance(details.get("ssh"), dict):
        info = parse_ssh_info(json.dumps(details["ssh"]))
        if info is not None:
            return info
    return parse_ssh_info(runner(["runpodctl", "ssh", "info", pod_id], check=False))


def wait_for_ssh(runner: Runner, pod_id: str, timeout: float, key: str | None,
                 poll: float = 15.0) -> dict | None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        info = pod_ssh(runner, pod_id)
        if info is not None and "ready" in runner(
            ssh_command({"ssh": info}, "echo ready", key), check=False
        ):
            return info
        time.sleep(poll)
    return None


def start_pod(runner: Runner, args, state: dict, shard: int, push_outputs: bool = False,
              attach: str | None = None) -> dict:
    """Create a pod for ``shard`` (falling back over GPU types/clouds) and start its job.

    With ``attach``, adopt that existing pod instead of creating one.
    """
    tried = []
    options = [(None, None, None)] if attach else state["candidates"]
    for gpu, cloud, _price in options:
        if attach:
            details = parse_json(runner(["runpodctl", "pod", "get", attach, "--output", "json"]))
            pod_id, gpu, cloud = attach, details.get("gpuId", "?"), "attached"
        else:
            try:
                pod_id = create_pod(runner, args, state, shard, gpu, cloud)
            except RuntimeError as error:  # usually no stock for this type/cloud
                reason = re.search(r'"error"\s*:\s*"([^"]+)"', str(error))
                tried.append(f"{gpu}/{cloud}: {reason.group(1) if reason else str(error)[-200:]}")
                print(f"shard {shard}: {tried[-1]}", flush=True)
                continue
        print(f"shard {shard}: {'attached' if attach else 'created'} {pod_id} ({gpu}, {cloud}); "
              f"waiting for direct SSH", flush=True)
        pod = {"id": pod_id, "gpu": gpu, "cloud": cloud,
               "created_utc": datetime.now(timezone.utc).isoformat()}
        if runner.dry_run:
            pod["ssh"] = {"user": "root", "host": "POD_IP", "port": 22}
        else:
            pod["ssh"] = wait_for_ssh(runner, pod_id, args.ssh_timeout, args.ssh_key)
            if pod["ssh"] is None:
                tried.append(f"{gpu}/{cloud}: no direct SSH within {args.ssh_timeout:.0f} s")
                if attach:
                    raise RuntimeError(f"attached pod {pod_id} has no direct SSH")
                print(f"shard {shard}: {tried[-1]}; deleting {pod_id}", flush=True)
                runner(["runpodctl", "pod", "delete", pod_id], check=False)
                continue
        details = parse_json(runner(["runpodctl", "pod", "get", pod_id, "--output", "json"],
                                    check=False) or "null") if not runner.dry_run else None
        if isinstance(details, dict):
            pod["cost_per_hr"] = details.get("costPerHr")
        deploy(runner, args, state, shard, pod, push_outputs)
        return pod
    raise RuntimeError(f"no pod could be started for shard {shard}:\n  " + "\n  ".join(tried))


def deploy(runner: Runner, args, state: dict, shard: int, pod: dict, push_outputs: bool) -> None:
    key = args.ssh_key
    runner(ssh_command(pod, "command -v rsync >/dev/null || "
                            "(apt-get update -qq && apt-get install -y -qq rsync >/dev/null)", key))
    runner(rsync_command(pod, str(args.bundle), f"{remote(pod)}:{REMOTE_WORK}/repo.bundle", key))
    runner(ssh_command(pod, (
        f"test -d {REMOTE_REPO}/.git || git clone -q --no-checkout {REMOTE_WORK}/repo.bundle {REMOTE_REPO}; "
        f"cd {REMOTE_REPO} && git fetch -q {REMOTE_WORK}/repo.bundle && git checkout -q {state['commit']}"
    ), key))
    if push_outputs:
        local = run_dir(state["run"]) / "grid"
        runner(ssh_command(pod, f"mkdir -p {REMOTE_REPO}/{state['outdir']}", key))
        runner(rsync_command(pod, f"{local}/", f"{remote(pod)}:{REMOTE_REPO}/{state['outdir']}/", key))
    runner(ssh_command(pod, shard_job(state, shard), key))


def make_bundle(runner: Runner, commit: str) -> Path:
    bundle = Path(tempfile.mkdtemp(prefix="asr-fleet-")) / "repo.bundle"
    runner(["git", "-C", str(ROOT), "bundle", "create", str(bundle), "--all"])
    return bundle


def committed_head(grid: str) -> str:
    """HEAD, after checking the tree is clean and the grid file is committed."""
    git = Runner()
    if not (ROOT / grid).exists():
        raise SystemExit(f"{grid} not found (give a path relative to the repository)")
    if git(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]).strip():
        raise SystemExit("uncommitted changes: commit first so pods run an exact commit")
    if not git(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", grid], check=False).strip():
        raise SystemExit(f"{grid} is not committed")
    return git(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).strip()


def command_launch(args, runner: Runner) -> None:
    if (run_dir(args.run) / "fleet.json").exists():
        raise SystemExit(f"run {args.run} already exists; use status/sync/relaunch")
    commit = committed_head(args.grid)
    pairs = candidates(args.gpu_types, args.clouds,
                       runner(["runpodctl", "gpu", "list", "--output", "json"], check=False),
                       args.max_price)
    state = {
        "run": args.run, "grid": args.grid, "outdir": grid_outdir(args.grid), "commit": commit,
        "shards": args.pods, "max_batch": args.max_batch, "model_batch_size": args.model_batch_size,
        "template": args.template, "candidates": pairs, "pods": {},
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    print(f"Launching {args.pods} pod(s) for {args.grid} at {commit[:10]}; "
          f"GPU preference: {', '.join(f'{g} ({c}, ${p}/h)' for g, c, p in pairs[:4])} ...",
          flush=True)
    if not args.yes and not runner.dry_run:
        if input("Pods bill until terminated. Continue? [y/N] ").strip().lower() != "y":
            raise SystemExit("aborted")
    args.bundle = make_bundle(runner, commit)
    if not runner.dry_run:
        save_state(state)
    if args.attach and len(args.attach) > args.pods:
        raise SystemExit("--attach lists more pods than --pods")
    for shard in range(args.pods):
        attach = args.attach[shard] if args.attach and shard < len(args.attach) else None
        try:
            state["pods"][str(shard)] = start_pod(runner, args, state, shard, attach=attach)
        finally:
            if not runner.dry_run:
                save_state(state)
    total = sum(float(pod.get("cost_per_hr") or 0) for pod in state["pods"].values())
    print(f"Started {len(state['pods'])} pod(s); about ${total:.2f}/h. "
          f"Next: fleet.py sync {args.run} --every 10 --terminate-done", flush=True)


def command_redeploy(args, runner: Runner) -> None:
    """Ship the current commit to a shard's existing pod and restart its job.

    For iterating on a failed pod without paying for a new one: bootstrap is
    idempotent and the shard's stages resume from their checkpoints.
    """
    state = load_state(args.run)
    pod = state["pods"][str(args.shard)]
    if pod.get("terminated"):
        raise SystemExit(f"shard {args.shard}'s pod was terminated; use relaunch")
    running = runner(ssh_command(pod, "pgrep -f 'run_shard.sh|bootstrap.sh' || true", args.ssh_key))
    if running.strip():
        raise SystemExit(f"shard {args.shard} is still running (pids {running.split()})")
    state["commit"] = committed_head(state["grid"])
    args.bundle = make_bundle(runner, state["commit"])
    runner(ssh_command(pod, f"rm -f {REMOTE_WORK}/status/FAILED {REMOTE_WORK}/status/DONE", args.ssh_key))
    deploy(runner, args, state, args.shard, pod, push_outputs=False)
    save_state(state)
    print(f"shard {args.shard}: redeployed {state['commit'][:10]} to {pod['id']}", flush=True)


def command_relaunch(args, runner: Runner) -> None:
    state = load_state(args.run)
    old = state["pods"].get(str(args.shard))
    if old and not old.get("terminated"):
        runner(["runpodctl", "pod", "delete", old["id"]], check=False)
    args.template = state["template"]
    args.bundle = make_bundle(runner, state["commit"])
    state["pods"][str(args.shard)] = start_pod(runner, args, state, args.shard, push_outputs=True)
    save_state(state)


# ------------------------------------------------------------ status / sync


def pod_status(runner: Runner, pod: dict, key: str | None) -> str:
    if pod.get("terminated"):
        return "terminated"
    output = runner(ssh_command(pod, f"cd {REMOTE_WORK}/status 2>/dev/null && "
                                     f"for f in DONE FAILED RUNNING; do test -f $f && "
                                     f"echo $f: $(cat $f) && break; done", key), check=False)
    return output.strip() or "bootstrapping (or unreachable)"


def sync_pod(runner: Runner, state: dict, shard: str, pod: dict, key: str | None) -> None:
    local = run_dir(state["run"])
    (local / "grid").mkdir(parents=True, exist_ok=True)
    (local / "pods" / f"shard_{shard}").mkdir(parents=True, exist_ok=True)
    runner(rsync_command(pod, f"{remote(pod)}:{REMOTE_REPO}/{state['outdir']}/", f"{local / 'grid'}/", key),
           check=False)
    runner(rsync_command(pod, f"{remote(pod)}:{REMOTE_WORK}/logs/", f"{local / 'pods' / f'shard_{shard}'}/", key),
           check=False)
    runner(rsync_command(pod, f"{remote(pod)}:{REMOTE_WORK}/status/",
                         f"{local / 'pods' / f'shard_{shard}' / 'status'}/", key), check=False)


def hours_since(iso: str) -> float:
    return (datetime.now(timezone.utc) - datetime.fromisoformat(iso)).total_seconds() / 3600


def terminate(runner: Runner, state: dict, shard: str, reason: str) -> None:
    pod = state["pods"][shard]
    runner(["runpodctl", "pod", "delete", pod["id"]], check=False)
    pod["terminated"] = reason
    pod["terminated_utc"] = datetime.now(timezone.utc).isoformat()
    save_state(state)
    print(f"shard {shard}: terminated {pod['id']} ({reason})", flush=True)


def command_status(args, runner: Runner) -> None:
    state = load_state(args.run)
    total = 0.0
    for shard, pod in sorted(state["pods"].items(), key=lambda item: int(item[0])):
        hours = hours_since(pod["created_utc"]) if not pod.get("terminated") else \
            (datetime.fromisoformat(pod["terminated_utc"]) - datetime.fromisoformat(pod["created_utc"])).total_seconds() / 3600
        cost = hours * float(pod.get("cost_per_hr") or 0)
        total += cost
        print(f"shard {shard}\t{pod['id']}\t{pod['gpu']} ({pod['cloud']})\t{hours:.2f} h\t"
              f"~${cost:.2f}\t{pod_status(runner, pod, args.ssh_key)}")
    print(f"total so far ~${total:.2f}")


def command_sync(args, runner: Runner) -> None:
    while True:
        state = load_state(args.run)
        active = {shard: pod for shard, pod in state["pods"].items() if not pod.get("terminated")}
        for shard, pod in sorted(active.items(), key=lambda item: int(item[0])):
            sync_pod(runner, state, shard, pod, args.ssh_key)
            status_dir = run_dir(args.run) / "pods" / f"shard_{shard}" / "status"
            if (status_dir / "DONE").exists() and args.terminate_done:
                terminate(runner, state, shard, "done")
            elif (status_dir / "FAILED").exists():
                print(f"shard {shard}: FAILED ({(status_dir / 'FAILED').read_text().strip()}); "
                      f"pod kept for inspection", flush=True)
            elif args.max_hours and hours_since(pod["created_utc"]) > args.max_hours:
                terminate(runner, state, shard, f"exceeded --max-hours {args.max_hours}")
        stamp = datetime.now().strftime("%H:%M")
        remaining = [s for s, p in load_state(args.run)["pods"].items() if not p.get("terminated")]
        print(f"[{stamp}] synced; {len(remaining)} pod(s) still running", flush=True)
        if not args.every or not remaining:
            return
        failed = [s for s in remaining
                  if (run_dir(args.run) / "pods" / f"shard_{s}" / "status" / "FAILED").exists()]
        if failed and len(failed) == len(remaining):
            print("every remaining pod has FAILED; stopping the sync loop", flush=True)
            return
        time.sleep(args.every * 60)


def command_logs(args, runner: Runner) -> None:
    pod = load_state(args.run)["pods"][str(args.shard)]
    print(runner(ssh_command(pod, f"tail -n {args.lines} {REMOTE_WORK}/logs/shard.log", args.ssh_key),
                 check=False))


def command_terminate(args, runner: Runner) -> None:
    state = load_state(args.run)
    shards = [str(args.shard)] if args.shard is not None else list(state["pods"])
    for shard in shards:
        if not state["pods"][shard].get("terminated"):
            terminate(runner, state, shard, "manual")


# --------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ssh-key", default=next(
        (str(path) for path in (Path.home() / ".ssh/id_ed25519", Path.home() / ".ssh/id_rsa") if path.exists()),
        None), help="private key registered with RunPod")
    sub = parser.add_subparsers(dest="command", required=True)
    launch = sub.add_parser("launch", help="create pods and start every shard")
    launch.add_argument("run", help="run name; state and synced output go to results/runpod/<run>/")
    launch.add_argument("--grid", required=True, help="committed grid TOML, relative to the repository")
    launch.add_argument("--pods", type=int, default=1, help="number of shards / pods")
    launch.add_argument("--gpu-types", nargs="+", default=DEFAULT_GPUS)
    launch.add_argument("--clouds", nargs="+", default=["COMMUNITY", "SECURE"],
                        choices=["COMMUNITY", "SECURE"])
    launch.add_argument("--template", default=DEFAULT_TEMPLATE, help="RunPod template id")
    launch.add_argument("--max-price", type=float, default=0.6, help="skip GPU types above this $/h")
    launch.add_argument("--volume-gb", type=int, default=40)
    launch.add_argument("--max-batch", type=int, default=64)
    launch.add_argument("--model-batch-size", type=int, default=32)
    launch.add_argument("--ssh-timeout", type=float, default=900)
    launch.add_argument("--attach", nargs="+", metavar="POD_ID",
                        help="adopt these running pods for shards 0.. instead of creating them")
    launch.add_argument("--yes", action="store_true", help="skip the cost confirmation prompt")
    launch.add_argument("--dry-run", action="store_true", help="print commands; create nothing")
    relaunch = sub.add_parser("relaunch", help="replace one shard's pod and resume from synced output")
    relaunch.add_argument("run")
    relaunch.add_argument("--shard", type=int, required=True)
    relaunch.add_argument("--volume-gb", type=int, default=40)
    relaunch.add_argument("--ssh-timeout", type=float, default=900)
    redeploy = sub.add_parser("redeploy", help="ship the current commit to a shard's pod and restart it")
    redeploy.add_argument("run")
    redeploy.add_argument("--shard", type=int, required=True)
    status = sub.add_parser("status", help="pods, elapsed time, cost and stage")
    status.add_argument("run")
    sync = sub.add_parser("sync", help="rsync outputs and logs home; optionally loop and clean up")
    sync.add_argument("run")
    sync.add_argument("--every", type=float, default=0, help="repeat every N minutes until all pods end")
    sync.add_argument("--terminate-done", action="store_true", help="delete pods once DONE has synced")
    sync.add_argument("--max-hours", type=float, default=0, help="delete pods older than this")
    logs = sub.add_parser("logs", help="tail a shard's log")
    logs.add_argument("run")
    logs.add_argument("--shard", type=int, required=True)
    logs.add_argument("-n", "--lines", type=int, default=40)
    terminate_parser = sub.add_parser("terminate", help="delete pods (all, or one shard)")
    terminate_parser.add_argument("run")
    terminate_parser.add_argument("--shard", type=int)
    return parser


def main(argv: list[str] | None = None, runner: Runner | None = None) -> None:
    args = build_parser().parse_args(argv)
    runner = runner or Runner(dry_run=getattr(args, "dry_run", False))
    {
        "launch": command_launch, "relaunch": command_relaunch, "redeploy": command_redeploy,
        "status": command_status,
        "sync": command_sync, "logs": command_logs, "terminate": command_terminate,
    }[args.command](args, runner)


if __name__ == "__main__":
    main()
