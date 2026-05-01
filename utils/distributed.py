import json
import os
import subprocess
import time
import psutil

import ray

def _get_sagemaker_hosts() -> list[str]:
    """Return ordered list of SageMaker host names from SM_HOSTS env var."""
    hosts_json = os.environ.get("SM_HOSTS", "[]")
    try:
        hosts = json.loads(hosts_json)
    except json.JSONDecodeError:
        hosts = []
    return sorted(hosts)


def _get_current_host() -> str:
    return os.environ.get("SM_CURRENT_HOST", "")


def _get_num_gpus() -> int:
    return int(os.environ.get("SM_NUM_GPUS", 0))

def _wait_for_ray_head(host: str, port: int, timeout: int = 120, interval: int = 5) -> None:
    """Block until the Ray head node's port is reachable."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=3):
                return
        except OSError:
            print(f"[Ray] Head not ready yet, retrying in {interval}s…")
            time.sleep(interval)
    raise TimeoutError(
        f"Ray head node {host}:{port} did not become reachable within {timeout}s"
    )

def init_ray_cluster(num_workers_per_node: int = 1) -> None:
    """Bootstrap a multi-node Ray cluster on SageMaker and call ray.init().

    SageMaker launches the training script on every instance simultaneously.
    We elect the first host (alphabetically) as the Ray head node, start
    `ray start --head` there, and `ray start --address` on all worker nodes.
    All nodes then connect via ray.init(address="auto").

    For single-instance jobs the function falls back to ray.init() as before.
    """
    hosts = _get_sagemaker_hosts()
    current_host = _get_current_host()
    num_gpus = _get_num_gpus()

    if len(hosts) <= 1:
        print("[Ray] Single-node mode — calling ray.init() directly.")
        ray.init(ignore_reinit_error=True)
        return

    head_host = hosts[0]
    ray_port = 6379
    ray_address = f"{head_host}:{ray_port}"

    num_cpus = psutil.cpu_count(logical=False) or 1
    resources_args = [
        f"--num-cpus={num_cpus}",
        f"--num-gpus={num_gpus}",
    ]

    if current_host == head_host:
        print(f"[Ray] Starting HEAD node on {current_host} (port {ray_port})")
        cmd = [
            "ray", "start", "--head",
            f"--port={ray_port}",
        ] + resources_args
        subprocess.run(cmd, check=True)
        print("[Ray] Head node started.")
    else:
        print(f"[Ray] Worker node {current_host} — waiting for head {head_host}…")
        _wait_for_ray_head(head_host, ray_port)

        print(f"[Ray] Starting WORKER node, connecting to {ray_address}")
        cmd = [
            "ray", "start",
            f"--address={ray_address}",
        ] + resources_args
        subprocess.run(cmd, check=True)
        print("[Ray] Worker node started.")

    print(f"[Ray] Calling ray.init(address='auto') on {current_host}")
    ray.init(address="auto", ignore_reinit_error=True)
    print(f"[Ray] Connected. Cluster resources: {ray.cluster_resources()}")