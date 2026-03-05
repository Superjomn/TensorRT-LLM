import asyncio
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import ray
except ModuleNotFoundError as e:
    e.msg = """Cannot import Ray. Please install 'ray' package to use ray orchestrator"""
    raise

from ray.util.placement_group import (PlacementGroupSchedulingStrategy,
                                      get_current_placement_group,
                                      placement_group)

from tensorrt_llm._ray_utils import unwrap_ray_errors
from tensorrt_llm._utils import nvtx_range_debug
from tensorrt_llm.logger import logger

from ..llmapi.utils import logger_debug
from .executor import GenerationExecutor
from .postproc_worker import PostprocWorkerConfig
from .ray_gpu_worker import RayGPUWorker, RayWorkerWrapper
from .request import GenerationRequest
from .result import GenerationResult
from .rpc_proxy_mixin import RpcExecutorMixin
from .utils import has_event_loop


def _dbg(msg: str):
    """Print debug message to stderr so it appears in ray-ray-job.err."""
    import datetime
    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[RAY_EXECUTOR_DEBUG {ts}] {msg}", file=sys.stderr, flush=True)

__all__ = [
    "RayExecutor",
]


class RayExecutor(RpcExecutorMixin, GenerationExecutor):

    def __init__(self,
                 worker_kwargs: Dict,
                 model_world_size: int,
                 postproc_worker_config: PostprocWorkerConfig,
                 is_llm_executor: bool,
                 tp_size=1):
        os.environ['RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES'] = '1'
        os.environ["RAY_DEDUP_LOGS"] = "0"  # for debug

        super().__init__(model_world_size, postproc_worker_config,
                         is_llm_executor)

        self.has_start_local_cluser = False
        runtime_env = {
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1"
            }
        }

        ray_init_args = {
            "include_dashboard": False,
            "namespace": "trtllm",
            "ignore_reinit_error": True,
            "runtime_env": runtime_env
        }

        try:
            if os.environ.get("TLLM_RAY_FORCE_LOCAL_CLUSTER", "0") != "1":
                try:
                    ray.init(address="auto", **ray_init_args)
                    logger.info(f"Attached to an existing Ray cluster.")
                except ConnectionError:
                    logger.info(f"Ray cluster not found, starting a new one.")

                if not ray.is_initialized():
                    ray.init(**ray_init_args)
                    self.has_start_local_cluser = True
            else:
                ray.init(address="local", **ray_init_args)
                self.has_start_local_cluser = True

            self.world_size = model_world_size
            self.tp_size = tp_size
            self.master_address = ray.util.get_node_ip_address()

            self.worker_kwargs = dict(
                **worker_kwargs,
                postproc_worker_config=postproc_worker_config,
                is_llm_executor=is_llm_executor)

            self.init_rpc_executor()
            # Inject the generated HMAC key into worker_kwargs for workers
            self.worker_kwargs['hmac_key'] = self.hmac_key
            self.worker_kwargs['rpc_addr'] = self.rpc_addr

            placement_config = getattr(self.worker_kwargs['llm_args'],
                                       'ray_placement_config', None)
            defer_workers_init = placement_config.defer_workers_init if placement_config else False

            if defer_workers_init:
                self.workers = [
                ]  # Placeholder, will be initialized in setup_async
                self._mainloop_started = False  # DO NOT start mainloop until after setup_engine_remote_async is called
            else:
                if not has_event_loop():
                    self.init_workers_sync()
                self.setup_engine_remote()
                self.setup_mainloop(tasks=[self._fetch_responses_loop_async],
                                    thread_name="ray_executor_main_loop")

        except Exception as e:
            self.shutdown()
            logger.error(f"Failed to initialize RayExecutor: {e}")
            raise e

    def create_workers(self, worker_cls, worker_kwargs):
        llm_args = worker_kwargs.get("llm_args")
        placement_config = getattr(llm_args, 'ray_placement_config',
                                   None) if llm_args else None

        # When set to be a fraction, it allows Ray to schedule
        # multiple actors on a single GPU for colocate use cases.
        num_gpus = float(os.getenv("TRTLLM_RAY_PER_WORKER_GPUS", "1.0"))
        if placement_config and placement_config.per_worker_gpu_share is not None:
            num_gpus = placement_config.per_worker_gpu_share

        logger.debug(f"{num_gpus=} for each worker.")

        # --- [RAY_EXECUTOR_DEBUG] Log cluster state before worker creation ---
        try:
            nodes = ray.nodes()
            _dbg(
                f"Cluster nodes: {len(nodes)}, "
                f"master_address={self.master_address}, world_size={self.world_size}, tp_size={self.tp_size}"
            )
            for i, node in enumerate(nodes):
                _dbg(
                    f"Node[{i}]: ip={node.get('NodeManagerAddress', 'N/A')}, "
                    f"alive={node.get('Alive', 'N/A')}, "
                    f"resources={node.get('Resources', {})}"
                )
            avail = ray.available_resources()
            cluster_res = ray.cluster_resources()
            _dbg(f"Cluster resources (total): {cluster_res}")
            _dbg(f"Available resources (free): {avail}")

            # Per-node GPU breakdown: show total vs available GPUs on each node
            _dbg("--- Per-node GPU breakdown ---")
            for i, node in enumerate(nodes):
                _node_ip = node.get('NodeManagerAddress', 'N/A')
                _node_id = node.get('NodeID', 'N/A')
                _total_res = node.get('Resources', {})
                _total_gpu = _total_res.get('GPU', 0)
                _total_cpu = _total_res.get('CPU', 0)
                _total_mem = _total_res.get('memory', 0)
                _total_obj = _total_res.get('object_store_memory', 0)
                _dbg(
                    f"Node[{i}] ip={_node_ip}, id={str(_node_id)[:16]}..., alive={node.get('Alive')}: "
                    f"total_GPU={_total_gpu}, total_CPU={_total_cpu}, "
                    f"memory={_total_mem/(1024**3):.1f}GB, obj_store={_total_obj/(1024**3):.1f}GB"
                )
            _dbg("--- End per-node GPU breakdown ---")
        except Exception as e:
            _dbg(f"WARNING: Failed to query cluster state: {e}")

        runtime_env = ray.runtime_env.RuntimeEnv()

        # --- [RAY_EXECUTOR_DEBUG] ITERATION 4 FIX: Filter out node-specific env vars ---
        # TRT-LLM was copying ALL 212 env vars from os.environ into runtime_env,
        # including node-specific Ray internals (RAY_RAYLET_PID, HOSTNAME, etc.)
        # that are WRONG for the remote node. This causes remote workers to connect
        # to the wrong raylet, resulting in immediate store.cc disconnection.
        _BLOCKLIST_ENV_PREFIXES = (
            "RAY_RAYLET_PID",
            "RAY_ADDRESS",
            "RAY_JOB_ID",
            "RAY_NODE_TYPE_HEAD",
            "HOSTNAME",
            "SLURMD_NODENAME",
            "SLURM_NODEID",
            "SLURM_LOCALID",
            "SLURM_GTIDS",
            "SLURM_PROCID",
            "SLURM_STEP_",
            "SLURM_TASK_",
            "SLURM_LAUNCH_NODE",
            "OMPI_",   # OpenMPI vars are node-specific
            "PMI_",    # PMI vars are node-specific
        )

        filtered_env = {}
        blocked = []
        for k, v in os.environ.items():
            if any(k.startswith(prefix) or k == prefix for prefix in _BLOCKLIST_ENV_PREFIXES):
                blocked.append(k)
            else:
                filtered_env[k] = v

        _dbg(f"Filtered env vars: kept {len(filtered_env)}, blocked {len(blocked)}: {blocked}")
        runtime_env["env_vars"] = filtered_env

        runtime_env["env_vars"].update({
            "TLLM_DISABLE_MPI": "1",
            "MASTER_ADDR": self.master_address,  # head-IP for NCCL/Gloo
        })

        # --- [RAY_EXECUTOR_DEBUG] Hypothesis test: confirm node-specific vars are blocked ---
        _dbg(f"RAY_RAYLET_PID in runtime_env: {'RAY_RAYLET_PID' in runtime_env.get('env_vars', {})}")
        _dbg(f"RAY_ADDRESS in runtime_env: {'RAY_ADDRESS' in runtime_env.get('env_vars', {})}")
        _dbg(f"HOSTNAME in runtime_env: {'HOSTNAME' in runtime_env.get('env_vars', {})}")
        _dbg(f"SLURMD_NODENAME in runtime_env: {'SLURMD_NODENAME' in runtime_env.get('env_vars', {})}")

        # --- [RAY_EXECUTOR_DEBUG] Log ALL env var keys being propagated ---
        _env = runtime_env["env_vars"]
        _dbg(f"runtime_env total env_vars count: {len(_env)}")
        _dbg(f"runtime_env env_var KEYS: {sorted(_env.keys())}")

        _debug_keys = [
            "CUDA_VISIBLE_DEVICES", "MASTER_ADDR", "MASTER_PORT", "RANK",
            "WORLD_SIZE", "RAY_ADDRESS", "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
            "TLLM_DISABLE_MPI", "NCCL_DEBUG", "LD_LIBRARY_PATH",
            "RAY_LOCAL_WORLD_SIZE", "TRTLLM_RAY_PER_WORKER_GPUS",
        ]
        for k in _debug_keys:
            if k in _env:
                val = _env[k] if k != "LD_LIBRARY_PATH" else f"(len={len(_env[k])})"
                _dbg(f"runtime_env[{k}]={val}")

        # --- [RAY_EXECUTOR_DEBUG] Check runtime_env serialization size ---
        try:
            _env_json = json.dumps({"env_vars": _env})
            _dbg(f"runtime_env serialized size: {len(_env_json)} bytes")
            # Also check for any env var with very long value
            _long_vars = [(k, len(v)) for k, v in _env.items() if len(v) > 500]
            if _long_vars:
                _dbg(f"Env vars with value > 500 chars: {_long_vars}")
        except Exception as e:
            _dbg(f"WARNING: Failed to serialize runtime_env: {e}")

        placement_groups, self.bundle_indices = self._get_placement_group(
            tp_size=self.tp_size, worker_kwargs=worker_kwargs)

        if isinstance(placement_groups, list):
            self.placement_group = None
        else:
            self.placement_group = placement_groups

        # --- [RAY_EXECUTOR_DEBUG] Log placement group details ---
        _dbg(
            f"placement_groups type={type(placement_groups).__name__}, "
            f"bundle_indices={self.bundle_indices}, num_gpus_per_worker={num_gpus}"
        )
        try:
            if isinstance(placement_groups, list):
                seen_pg_ids = set()
                for idx, pg in enumerate(placement_groups):
                    pg_id = id(pg)
                    if pg_id not in seen_pg_ids:
                        seen_pg_ids.add(pg_id)
                        from ray.util import placement_group_table
                        pg_info = placement_group_table(pg)
                        _dbg(
                            f"PG(id={pg_id}): state={pg_info.get('state', 'N/A')}, "
                            f"bundles={pg_info.get('bundles', 'N/A')}, "
                            f"bundles_to_node_id={pg_info.get('bundles_to_node_id', 'N/A')}"
                        )
            else:
                from ray.util import placement_group_table
                pg_info = placement_group_table(placement_groups)
                _dbg(f"PG info: {pg_info}")
        except Exception as e:
            _dbg(f"WARNING: Failed to inspect PG: {e}")

        # --- [RAY_EXECUTOR_DEBUG] Resolve node assignment per bundle ---
        _bundle_to_node = {}
        try:
            if isinstance(placement_groups, list):
                seen_pgs = {}
                for rank_idx in range(self.world_size):
                    pg = placement_groups[rank_idx]
                    pg_key = id(pg)
                    if pg_key not in seen_pgs:
                        from ray.util import placement_group_table
                        pg_info = placement_group_table(pg)
                        seen_pgs[pg_key] = pg_info.get('bundles_to_node_id', {})
                    b2n = seen_pgs[pg_key]
                    bundle_idx = self.bundle_indices[rank_idx]
                    # bundles_to_node_id may use string keys
                    node_id = b2n.get(bundle_idx, b2n.get(str(bundle_idx), 'unknown'))
                    _bundle_to_node[rank_idx] = node_id
            else:
                from ray.util import placement_group_table
                pg_info = placement_group_table(placement_groups)
                b2n = pg_info.get('bundles_to_node_id', {})
                for rank_idx in range(self.world_size):
                    bundle_idx = self.bundle_indices[rank_idx]
                    node_id = b2n.get(bundle_idx, b2n.get(str(bundle_idx), 'unknown'))
                    _bundle_to_node[rank_idx] = node_id
            _dbg(f"Bundle-to-node mapping: { {r: n[:12]+'...' if len(str(n))>12 else n for r, n in _bundle_to_node.items()} }")
            # Group by node
            _node_ranks = {}
            for r, n in _bundle_to_node.items():
                _node_ranks.setdefault(n, []).append(r)
            for n, ranks in _node_ranks.items():
                _short = n[:16] + '...' if len(str(n)) > 16 else n
                _dbg(f"Node {_short}: ranks={ranks} ({len(ranks)} workers)")
        except Exception as e:
            _dbg(f"WARNING: Failed to resolve bundle-to-node: {e}")

        # Identify which node is "local" (same as TRTLLMHttpServer/master)
        _local_node_id = None
        try:
            _my_node_ip = ray.util.get_node_ip_address()
            for node in ray.nodes():
                if node.get('NodeManagerAddress') == _my_node_ip and node.get('Alive'):
                    _local_node_id = node.get('NodeID')
                    break
            _dbg(f"Local node (TRTLLMHttpServer): ip={_my_node_ip}, node_id={_local_node_id[:16] if _local_node_id else 'unknown'}...")
        except Exception as e:
            _dbg(f"WARNING: Failed to identify local node: {e}")

        # --- [RAY_EXECUTOR_DEBUG] Check for PG overlap / GPU double-booking ---
        try:
            _dbg("--- Placement Group overlap check ---")
            # Get the current placement group (from verl's outer context)
            _outer_pg = get_current_placement_group()
            if _outer_pg is not None:
                from ray.util import placement_group_table
                _outer_info = placement_group_table(_outer_pg)
                _dbg(
                    f"OUTER PG (from verl context): state={_outer_info.get('state', 'N/A')}, "
                    f"bundles={_outer_info.get('bundles', 'N/A')}, "
                    f"bundles_to_node_id={_outer_info.get('bundles_to_node_id', 'N/A')}"
                )
                # Check if outer PG nodes overlap with TRT-LLM PG nodes
                _outer_nodes = set(_outer_info.get('bundles_to_node_id', {}).values())
                _trtllm_nodes = set(_bundle_to_node.values())
                _overlap = _outer_nodes & _trtllm_nodes
                if _overlap:
                    _dbg(f"WARNING: OUTER PG and TRT-LLM PG share nodes: {[n[:16]+'...' for n in _overlap]}")
                    # Check GPU counts on overlapping nodes
                    for _node_id in _overlap:
                        _outer_gpus = sum(
                            b.get('GPU', 0) for b_idx, b in enumerate(_outer_info.get('bundles', []))
                            if _outer_info.get('bundles_to_node_id', {}).get(str(b_idx)) == _node_id
                            or _outer_info.get('bundles_to_node_id', {}).get(b_idx) == _node_id
                        )
                        _trtllm_ranks_on_node = [r for r, n in _bundle_to_node.items() if n == _node_id]
                        _dbg(
                            f"Overlapping node {str(_node_id)[:16]}...: "
                            f"outer_PG_GPUs={_outer_gpus}, "
                            f"trtllm_ranks={_trtllm_ranks_on_node} (requesting {num_gpus} GPU each = {len(_trtllm_ranks_on_node)*num_gpus} total)"
                        )
                else:
                    _dbg(f"No node overlap between OUTER PG and TRT-LLM PG")
            else:
                _dbg(f"No OUTER placement group in current context")

            # Also check available resources RIGHT NOW on each node
            _dbg("--- Available resources per node (post-PG) ---")
            for node in ray.nodes():
                if not node.get('Alive'):
                    continue
                _node_ip = node.get('NodeManagerAddress', 'N/A')
                _node_id = node.get('NodeID', 'N/A')
                # ray.available_resources() is cluster-wide; we need per-node
                # Use node's Resources (total) for reference
                _total_gpu = node.get('Resources', {}).get('GPU', 0)
                _dbg(f"Node ip={_node_ip}, id={str(_node_id)[:16]}...: total_GPU={_total_gpu}")
            _dbg("--- End PG overlap check ---")
        except Exception as e:
            _dbg(f"WARNING: PG overlap check failed: {e}")

        # --- [RAY_EXECUTOR_DEBUG] nvidia-smi on local node before worker creation ---
        try:
            import subprocess
            _nvsmi_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            _dbg(f"nvidia-smi on LOCAL node ({_my_node_ip}):")
            for _line in _nvsmi_result.stdout.strip().split('\n'):
                _dbg(f"  GPU {_line.strip()}")
            if _nvsmi_result.returncode != 0:
                _dbg(f"nvidia-smi stderr: {_nvsmi_result.stderr.strip()}")
        except Exception as e:
            _dbg(f"nvidia-smi failed: {e}")

        # --- [RAY_EXECUTOR_DEBUG] Try nvidia-smi on REMOTE nodes via Ray remote ---
        _first_remote_done = False
        try:
            @ray.remote(num_cpus=0.01)
            def _remote_nvidia_smi():
                import subprocess as _sp
                import socket as _sk
                _hostname = _sk.gethostname()
                _r = _sp.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,gpu_uuid",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5
                )
                return _hostname, _r.stdout.strip(), _r.returncode, _r.stderr.strip()

            # Run nvidia-smi on each unique node
            _node_ips = set()
            for node in ray.nodes():
                if node.get('Alive'):
                    _node_ips.add(node.get('NodeManagerAddress'))
            _dbg(f"Running nvidia-smi on all {len(_node_ips)} nodes via Ray remote...")
            _nvsmi_refs = [_remote_nvidia_smi.remote() for _ in range(len(_node_ips))]
            _nvsmi_results = ray.get(_nvsmi_refs, timeout=15)
            _seen_hosts = set()
            for _hostname, _stdout, _rc, _stderr_out in _nvsmi_results:
                if _hostname not in _seen_hosts:
                    _seen_hosts.add(_hostname)
                    _dbg(f"nvidia-smi on {_hostname}:")
                    for _line in _stdout.split('\n'):
                        _dbg(f"  GPU {_line.strip()}")
                    if _rc != 0:
                        _dbg(f"  stderr: {_stderr_out}")
        except Exception as e:
            _dbg(f"Remote nvidia-smi failed: {e}")

        # --- [RAY_EXECUTOR_DEBUG] Check available resources RIGHT BEFORE worker creation ---
        try:
            _avail_now = ray.available_resources()
            _dbg(f"Available resources RIGHT BEFORE worker creation: {_avail_now}")
        except Exception as e:
            _dbg(f"WARNING: Failed to check available resources: {e}")

        self.workers = []
        _t_start_all = time.monotonic()
        for rank in range(self.world_size):
            pg = placement_groups[rank] if isinstance(
                placement_groups, list) else placement_groups

            # Determine if this worker goes to a remote node
            _target_node = _bundle_to_node.get(rank, 'unknown')
            _is_remote = (_local_node_id is not None and _target_node != _local_node_id)
            _node_label = "REMOTE" if _is_remote else "LOCAL"

            _dbg(
                f"Creating worker rank={rank}, "
                f"pg_id={id(pg)}, bundle_index={self.bundle_indices[rank]}, num_gpus={num_gpus}, "
                f"target_node={str(_target_node)[:16]}..., {_node_label}"
            )

            # Log extra detail for the FIRST remote worker
            if _is_remote and not _first_remote_done:
                _first_remote_done = True
                _dbg(f"=== FIRST REMOTE WORKER (rank={rank}) ===")
                try:
                    _avail_at_remote = ray.available_resources()
                    _dbg(f"Available resources at first remote creation: {_avail_at_remote}")
                except Exception:
                    pass
                # Log the placement group scheduling strategy details
                _dbg(
                    f"Scheduling: pg_id={id(pg)}, bundle_index={self.bundle_indices[rank]}, "
                    f"num_gpus={num_gpus}, runtime_env keys={sorted(runtime_env.get('env_vars', {}).keys()) if isinstance(runtime_env, dict) else 'RuntimeEnv obj'}"
                )

            # Stagger remote-node worker creation to reduce contention
            if _is_remote and rank > 0:
                _stagger_delay = float(os.environ.get("TRTLLM_RAY_STAGGER_DELAY", "0.5"))
                if _stagger_delay > 0:
                    _dbg(f"Staggering remote worker rank={rank} by {_stagger_delay}s")
                    time.sleep(_stagger_delay)

            _t_before = time.monotonic()
            worker = RayWorkerWrapper.options(
                num_gpus=num_gpus,
                runtime_env=runtime_env,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=self.bundle_indices[rank],
                )).remote(worker_cls, worker_kwargs, self.world_size, rank)
            _t_after = time.monotonic()
            _dbg(f"Worker rank={rank} actor created in {_t_after - _t_before:.3f}s (actor_id={worker._actor_id.hex() if hasattr(worker, '_actor_id') else 'N/A'})")
            self.workers.append(worker)

        _t_end_all = time.monotonic()
        _dbg(f"All {self.world_size} worker actors created in {_t_end_all - _t_start_all:.3f}s (not yet ready)")

    def init_workers_sync(self):
        self.create_workers(RayGPUWorker, self.worker_kwargs)
        _dbg(f"init_workers_sync: waiting for {len(self.workers)} workers...")
        _t_sync_start = time.monotonic()
        try:
            ray.get(self._get_worker_ready_futures())
        except ray.exceptions.ActorDiedError as e:
            _dbg(f"ActorDiedError in init_workers_sync after {time.monotonic()-_t_sync_start:.1f}s: {e}")
            # Check each worker's state
            for i, w in enumerate(self.workers):
                try:
                    ray.get(w.__ray_ready__.remote(), timeout=2.0)
                    _dbg(f"Worker rank={i} is alive")
                except Exception as we:
                    _dbg(f"Worker rank={i} DEAD/UNREACHABLE: {type(we).__name__}: {we}")
            # Check cluster state at time of failure
            try:
                _nodes = ray.nodes()
                for _n in _nodes:
                    _dbg(f"Node at failure: ip={_n.get('NodeManagerAddress')}, alive={_n.get('Alive')}, resources={_n.get('Resources', {})}")
            except Exception:
                pass
            raise RuntimeError("RayGPUWorker died during initialization") from e
        _dbg(f"All workers ready (sync) in {time.monotonic()-_t_sync_start:.1f}s. Setting up TCP store...")
        port = self.call_all_ray_workers("setup_tcp_store",
                                         leader_only=True,
                                         async_call=False)[0]
        _dbg(f"TCP store port={port}. Setting up distributed env...")
        self.call_all_ray_workers("setup_distributed_env_and_worker",
                                  leader_only=False,
                                  async_call=False,
                                  port=port)
        _dbg(f"init_workers_sync complete.")

    async def init_workers_async(self):
        self.create_workers(RayGPUWorker, self.worker_kwargs)
        _dbg(f"init_workers_async: waiting for {len(self.workers)} workers to be ready...")

        # --- [RAY_EXECUTOR_DEBUG] Map future -> rank for identification ---
        _t_wait_start = time.monotonic()
        try:
            ready_futures = self._get_worker_ready_futures()
            _future_to_rank = {}
            for i, f in enumerate(ready_futures):
                _future_to_rank[f] = i
            _dbg(f"init_workers_async: {len(ready_futures)} ready futures created, starting ray.wait loop")

            # Use ray.wait to identify which worker(s) fail first
            remaining = list(ready_futures)
            ready_count = 0
            while remaining:
                _dbg(f"ray.wait: {ready_count} ready, {len(remaining)} remaining, elapsed={time.monotonic()-_t_wait_start:.1f}s")
                ready, remaining = ray.wait(remaining, num_returns=1, timeout=30.0)
                if not ready:
                    _dbg(
                        f"Timeout (30s) waiting for workers. "
                        f"{ready_count}/{len(self.workers)} ready, {len(remaining)} remaining, "
                        f"total_elapsed={time.monotonic()-_t_wait_start:.1f}s"
                    )
                    # Check each worker's state individually
                    for i, w in enumerate(self.workers):
                        try:
                            _aid = w._actor_id.hex() if hasattr(w, '_actor_id') else "no_id"
                            _dbg(f"Worker rank={i} actor_id={_aid}")
                        except Exception:
                            pass
                    # Also check cluster resources
                    try:
                        _avail = ray.available_resources()
                        _dbg(f"Available resources during wait: {_avail}")
                    except Exception:
                        pass
                    continue
                try:
                    # Identify which rank just became ready
                    _ready_rank = _future_to_rank.get(ready[0], "unknown")
                    await asyncio.gather(*ready)
                    ready_count += 1
                    _dbg(f"Worker rank={_ready_rank} ready ({ready_count}/{len(self.workers)}, elapsed={time.monotonic()-_t_wait_start:.1f}s)")
                except ray.exceptions.ActorDiedError as e:
                    _failed_rank = _future_to_rank.get(ready[0], "unknown")
                    _dbg(
                        f"Worker rank={_failed_rank} DIED during init! "
                        f"ready_count={ready_count}/{len(self.workers)}, "
                        f"elapsed={time.monotonic()-_t_wait_start:.1f}s, "
                        f"error_type={type(e).__name__}"
                    )
                    _dbg(f"ActorDiedError details: {e}")
                    # Extract actor death cause if available
                    if hasattr(e, 'actor_id'):
                        _dbg(f"Dead actor_id={e.actor_id}")
                    if hasattr(e, 'error_msg'):
                        _dbg(f"Dead actor error_msg={e.error_msg}")

                    # Try to get more details about each worker
                    for i, w in enumerate(self.workers):
                        try:
                            ray.get(w.__ray_ready__.remote(), timeout=2.0)
                            _dbg(f"Worker rank={i} is alive")
                        except ray.exceptions.ActorDiedError as we:
                            _dbg(f"Worker rank={i} DEAD: {we}")
                        except ray.exceptions.GetTimeoutError:
                            _dbg(f"Worker rank={i} timeout (may be pending)")
                        except Exception as we:
                            _dbg(f"Worker rank={i} check error: {type(we).__name__}: {we}")

                    # Check cluster state at time of failure
                    try:
                        _nodes = ray.nodes()
                        for _n in _nodes:
                            _dbg(f"Node at failure: ip={_n.get('NodeManagerAddress')}, alive={_n.get('Alive')}, resources={_n.get('Resources', {})}")
                    except Exception:
                        pass

                    raise RuntimeError("RayGPUWorker died during initialization") from e
        except ray.exceptions.ActorDiedError as e:
            _dbg(f"ActorDiedError in init_workers_async (outer): {e}")
            raise RuntimeError("RayGPUWorker died during initialization") from e

        _dbg(f"All workers ready. Setting up TCP store...")
        port = (await asyncio.gather(*self.call_all_ray_workers(
            "setup_tcp_store", leader_only=True, async_call=True)))[0]
        _dbg(f"TCP store on port={port}. Setting up distributed env...")
        await asyncio.gather(
            *self.call_all_ray_workers("setup_distributed_env_and_worker",
                                       leader_only=False,
                                       async_call=True,
                                       port=port))
        _dbg(f"init_workers_async complete.")

    @unwrap_ray_errors()
    def call_all_ray_workers(self, func: str, leader_only: bool,
                             async_call: bool, *args, **kwargs):
        workers = (self.workers[0], ) if leader_only else self.workers
        if async_call:
            return [
                getattr(worker, func).remote(*args, **kwargs)
                for worker in workers
            ]
        else:
            return ray.get([
                getattr(worker, func).remote(*args, **kwargs)
                for worker in workers
            ])

    @unwrap_ray_errors()
    def collective_rpc(self,
                       method: str,
                       args: tuple = (),
                       kwargs: Optional[dict] = None,
                       non_block: bool = False,
                       unique_reply_rank: Optional[int] = None) -> list[Any]:
        workers = (self.workers[unique_reply_rank],
                   ) if unique_reply_rank is not None else self.workers
        kwargs = kwargs or {}

        refs = []
        for w in workers:
            try:
                refs.append(getattr(w, method).remote(*args, **kwargs))
            except AttributeError:
                # Here worker is the RayWorkerWrapper.
                # For extended worker methods, we need to use call_worker_method since
                # Ray actor doesn't work with __getattr__ delegation.
                refs.append(w.call_worker_method.remote(method, *args,
                                                        **kwargs))
        return refs if non_block else ray.get(refs)

    @unwrap_ray_errors()
    async def collective_rpc_async(
            self,
            method: str,
            args: tuple = (),
            kwargs: Optional[dict] = None,
            unique_reply_rank: Optional[int] = None) -> list[Any]:
        refs = self.collective_rpc(method,
                                   args,
                                   kwargs,
                                   non_block=True,
                                   unique_reply_rank=unique_reply_rank)
        return await asyncio.gather(*refs)

    def submit(self, request: "GenerationRequest") -> "GenerationResult":
        """
        Low-level API to the executor. Return a "future" GenerationResult
        which can be waited. Forwards the request to the workers through RPC.
        """
        request.set_id(self._get_next_client_id())
        logprob_params = self._get_logprob_params(request)

        with nvtx_range_debug("rpc_submit"):
            self.rpc_client.submit(request).remote(need_response=False)

        result = GenerationResult(
            request,
            background_error_handler=self._handle_background_error,
            executor=self,
            disaggregated_params=request.disaggregated_params,
            logprob_params=logprob_params)
        self._results[request.id] = result

        return result

    def start(self):
        pass

    def setup_engine_remote(self):
        return self.collective_rpc("setup_engine", non_block=False)

    async def setup_engine_remote_async(self):
        """Async version of setup_engine_remote for use after async worker initialization."""
        if not self.workers or len(self.workers) == 0:
            raise RuntimeError(
                "Workers must be initialized before calling setup_engine_remote_async"
            )

        # Setup engine on all workers
        result = await self.collective_rpc_async("setup_engine")
        logger.info("setup_engine_remote_async finished")

        # Now that engine is set up, start the mainloop for fetching responses
        if hasattr(self, '_mainloop_started') and not self._mainloop_started:
            logger.info("Starting mainloop after engine setup")
            self.setup_mainloop(tasks=[self._fetch_responses_loop_async],
                                thread_name="ray_executor_main_loop")
            self._mainloop_started = True

        return result

    def report_device_ids(self) -> list[str]:
        gpu_ids = self.call_all_ray_workers("report_device_id",
                                            leader_only=False,
                                            async_call=False)
        return sorted(gpu_ids)

    def abort_request(self, request_id: int) -> None:
        self.call_all_ray_workers("abort_request",
                                  leader_only=True,
                                  async_call=False,
                                  request_id=request_id)

    def shutdown(self):
        if hasattr(self, '_shutdown_event') and self._shutdown_event.is_set():
            return
        if hasattr(self, '_shutdown_event'):
            self._shutdown_event.set()

        logger_debug(f"Shutting down RayExecutor", color="yellow")

        if hasattr(self, 'main_loop') and self.main_loop and hasattr(
                self, 'main_loop_task_obj') and self.main_loop_task_obj:
            logger_debug("Cancelling main loop task.", color="yellow")
            try:
                self.main_loop.call_soon_threadsafe(
                    self.main_loop_task_obj.cancel)
            except Exception as e:
                logger_debug(f"Error cancelling main loop task: {e}",
                             color="yellow")

            if hasattr(self, 'main_loop_thread'):
                self.main_loop_thread.join()

        # Then, shutdown the workers
        if hasattr(self, 'workers') and self.workers is not None:
            try:
                shutdown_refs = [
                    worker.shutdown.remote() for worker in self.workers
                ]
                # Add timeout to prevent indefinite hanging
                ray.get(shutdown_refs, timeout=30.0)
            except ray.exceptions.GetTimeoutError:
                logger.warning(
                    "Timeout waiting for workers to shutdown after 30 seconds")
            except Exception as e:
                logger.warning(f"Error shutting down: {e}")

        if hasattr(self, 'rpc_client') and self.rpc_client is not None:
            try:
                self.rpc_client.close()
            except Exception as e:
                logger_debug(f"Suppressed error during RPC client close: {e}")

        self.workers = None
        if hasattr(self,
                   "placement_group") and self.placement_group is not None:
            # Only remove placement group if Ray is still initialized
            # to avoid triggering auto_init_ray() during program exit
            if ray.is_initialized():
                ray.util.remove_placement_group(self.placement_group)
            self.placement_group = None
        self.bundle_indices = None

        if self.has_start_local_cluser and ray.is_initialized():
            logger.debug("Shutting down Ray cluster")
            ray.shutdown()

    def _get_worker_ready_futures(self):
        return [worker.__ray_ready__.remote() for worker in self.workers]

    def _get_placement_group(
            self,
            tp_size: int,
            worker_kwargs: Dict = None) -> Tuple[Any, List[int]]:
        """
        Either use the existing placement group from driver script (e.g., in the case of RL FW integration),
        or create a default PACK placement group where each bundle has tp_size GPUs.
         - When tp_size ≤ GPUs per node, keep one TP group per node.
         - When tp_size >  GPUs per node, allow a TP group span nodes.
         - rank 0 must be put on the driver node

        Returns:
            Tuple of (placement_group(s), bundle_indices)
            - placement_group(s) can be a single PlacementGroup or a List[PlacementGroup]
            - bundle_indices is always a List[int]
        """
        llm_args = worker_kwargs.get("llm_args") if worker_kwargs else None

        placement_config = getattr(llm_args, 'ray_placement_config',
                                   None) if llm_args else None
        if placement_config and placement_config.placement_groups is not None:
            total_workers = sum(
                len(indices)
                for indices in placement_config.placement_bundle_indices)
            if total_workers != self.world_size:
                raise ValueError(
                    f"Total bundle indices ({total_workers}) must equal world_size ({self.world_size})"
                )

            logger.info(
                f"Creating {self.world_size} workers with external placement groups"
            )

            # --- [RAY_EXECUTOR_DEBUG] Log external PG details ---
            try:
                from ray.util import placement_group_table
                for i, (pg, indices) in enumerate(zip(
                        placement_config.placement_groups,
                        placement_config.placement_bundle_indices)):
                    pg_info = placement_group_table(pg)
                    _dbg(
                        f"External PG[{i}]: state={pg_info.get('state', 'N/A')}, "
                        f"bundle_count={len(pg_info.get('bundles', []))}, "
                        f"bundles={pg_info.get('bundles', 'N/A')}, "
                        f"bundles_to_node_id={pg_info.get('bundles_to_node_id', 'N/A')}, "
                        f"assigned_indices={indices}"
                    )
            except Exception as e:
                _dbg(f"WARNING: Failed to inspect external PGs: {e}")

            flat_pgs = []
            flat_indices = []
            for pg, indices in zip(placement_config.placement_groups,
                                   placement_config.placement_bundle_indices):
                for idx in indices:
                    flat_pgs.append(pg)
                    flat_indices.append(idx)

            _dbg(
                f"External PGs flattened: "
                f"{len(flat_pgs)} entries, flat_indices={flat_indices}"
            )

            return flat_pgs, flat_indices

        bundle_indices = os.getenv("TRTLLM_RAY_BUNDLE_INDICES", None)

        if bundle_indices:
            pg = get_current_placement_group()
            if pg is not None:
                bundle_indices = list(map(int, bundle_indices.split(",")))
                assert len(bundle_indices) == self.world_size, (
                    f"Need {self.world_size} bundle indices for world_size, got {bundle_indices=}"
                )
                assert len(set(bundle_indices)) == len(bundle_indices), \
                    f"TRTLLM_RAY_BUNDLE_INDICES cannot have duplicate values, but got {bundle_indices=}."

                assert max(bundle_indices) < len(pg.bundle_specs), \
                    f"{bundle_indices=} out of range for PG with {len(pg.bundle_specs)} bundles"

                logger.info(
                    f"Found existing placement group {pg.bundle_specs=}. {bundle_indices=}"
                )

                # TODO: need to ping TP group onto the same node for RL FW integration case

                return pg, bundle_indices
            else:
                logger.warning(
                    f"Ignoring TRTLLM_RAY_BUNDLE_INDICES={bundle_indices} because no global placement group is found."
                )

        if self.world_size % tp_size:
            raise ValueError("world_size must be a multiple of tp_size")

        head_tag = f"node:{self.master_address}"
        nodes = ray.nodes()
        gpus_per_node = int(nodes[0]["Resources"].get(
            "GPU", 0))  # assume symmetric across nodes

        bundle_cpu = bundle_gpu = min(tp_size, gpus_per_node)

        bundles, bundle_indices = [], []
        current = 0
        for rank in range(self.world_size):
            if current == 0:
                bundle = {"GPU": bundle_gpu, "CPU": bundle_cpu}
                if len(bundles) == 0:
                    bundle[head_tag] = 0.01  # to force placement on head node
                bundles.append(bundle)

            bundle_indices.append(len(bundles) - 1)
            current = (current + 1) % bundle_gpu

        strategy = "PACK"
        logger.debug(
            f"[Strategy={strategy}] Bundles: {bundles} for tp_size: {tp_size} and world_size: {self.world_size}"
        )
        pg = placement_group(bundles, strategy=strategy)

        return pg, bundle_indices

    @property
    def enable_postprocess_parallel(self) -> bool:
        ret = super().enable_postprocess_parallel
        assert ret == False, "Postprocess parallel is not supported in RayExecutor"
        return ret
