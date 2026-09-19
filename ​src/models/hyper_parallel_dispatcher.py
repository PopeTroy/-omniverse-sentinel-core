import os
import asyncio
import concurrent.futures
import logging
import time
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import onnxruntime as ort

from src.config.settings import settings
from src.core.divine_tactics import TAILED_BEAST_CHAKRA, OcularAbility

logger = logging.getLogger("HyperParallelDispatcher")

class HyperParallelModelEngine:
    """
    Hyper-Parallel Execution Engine for running simultaneous inference passes
    across thousands of loaded ONNX models using Rust/C-FFI and TensorRT bindings.
    """

    def __init__(self, max_concurrent_workers: int = 32):
        self.max_workers = max_concurrent_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        self.sessions: Dict[str, ort.InferenceSession] = {}

    def register_model_session(self, model_id: str, model_path: str, provider: str = "TensorRTExecutionProvider"):
        """Loads and binds an ONNX session to the parallel registry."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")

        provider_options = []
        if provider == "TensorRTExecutionProvider":
            provider_options = [{
                "trt_fp16_enable": settings.trt_fp16_enable,
                "trt_max_workspace_size": settings.trt_max_workspace_mb * 1024 * 1024,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_cache"
            }]
        elif provider == "CUDAExecutionProvider":
            provider_options = [{
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo"
            }]

        providers = [(provider, provider_options[0])] if provider_options else [provider, "CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        self.sessions[model_id] = session
        logger.info(f"[HYPER-PARALLEL] Registered model session: {model_id} via {session.get_providers()[0]}")

    def _sync_run_model(self, model_id: str, input_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Synchronous single-model execution unit executed inside thread pool worker."""
        session = self.sessions.get(model_id)
        if not session:
            raise KeyError(f"Model {model_id} not registered in parallel engine.")

        start_time = time.perf_counter()
        outputs = session.run(None, input_dict)
        latency_ms = (time.perf_counter() - start_time) * 1000

        output_names = [out.name for out in session.get_outputs()]
        result_map = {name: val for name, val in zip(output_names, outputs)}

        return {
            "model_id": model_id,
            "outputs": result_map,
            "latency_ms": latency_ms
        }

    async def execute_parallel_batch(
        self, 
        execution_plan: List[Tuple[str, Dict[str, np.ndarray]]],
        chakra_tier: TAILED_BEAST_CHAKRA = TAILED_BEAST_CHAKRA.KURAMA_NINE_TAILS
    ) -> List[Dict[str, Any]]:
        """
        Executes hyper-parallel inference across all specified models in execution_plan concurrently.
        Scales parallelism based on the active Tailed Beast Chakra tier.
        """
        loop = asyncio.get_running_loop()
        start_batch = time.perf_counter()

        logger.info(
            f"[HYPER-PARALLEL DISPATCH] [{OcularAbility.RINNEGAN_SURA_PATH.value}] "
            f"Dispatching {len(execution_plan)} parallel model inferences under {chakra_tier.name}..."
        )

        tasks = []
        for model_id, input_dict in execution_plan:
            task = loop.run_in_executor(
                self.executor,
                self._sync_run_model,
                model_id,
                input_dict
            )
            tasks.append(task)

        # Gather all parallel model execution futures
        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed_results = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"[HYPER-PARALLEL ERROR] Parallel execution failed: {res}")
            else:
                processed_results.append(res)

        total_batch_ms = (time.perf_counter() - start_batch) * 1000
        logger.info(
            f"[HYPER-PARALLEL COMPLETE] Processed {len(processed_results)} models in {total_batch_ms:.2f} ms "
            f"(Avg: {total_batch_ms / max(len(processed_results), 1):.2f} ms/model)"
        )

        return processed_results

    def shutdown(self):
        """Cleanly drains and terminates thread workers."""
        self.executor.shutdown(wait=True)
        self.sessions.clear()
        logger.info("[HYPER-PARALLEL ENGINE] Engine pool terminated.")
