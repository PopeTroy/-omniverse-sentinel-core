import os
import time
import numpy as np
import onnxruntime as ort
from typing import Dict, Any, List

class TensorRTAutoTuner:
    """Profiles and optimizes ONNX execution providers for maximum GPU throughput."""
    
    def __init__(self, model_paths: List[str]):
        self.model_paths = model_paths

    def profile_and_tune(self, sample_input_shape: tuple) -> Dict[str, Any]:
        results = {}
        dummy_input = np.random.randn(*sample_input_shape).astype(np.float32)

        for path in self.model_paths:
            model_id = os.path.basename(path)
            
            # Provider options optimized for high-parallelism throughput
            providers = [
                ('TensorRTExecutionProvider', {
                    'trt_fp16_enable': True,
                    'trt_max_workspace_size': 2 << 30,  # 2GB Memory Budget per Model
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': './trt_cache'
                }),
                'CUDAExecutionProvider'
            ]
            
            sess = ort.InferenceSession(path, providers=providers)
            input_name = sess.get_inputs()[0].name
            
            # Warmup
            for _ in range(10):
                _ = sess.run(None, {input_name: dummy_input})
                
            # Benchmark Latency
            start_time = time.perf_counter()
            runs = 100
            for _ in range(runs):
                _ = sess.run(None, {input_name: dummy_input})
            end_time = time.perf_counter()

            avg_latency_ms = ((end_time - start_time) / runs) * 1000.0
            results[model_id] = {
                "avg_latency_ms": avg_latency_ms,
                "provider": sess.get_providers()[0]
            }
            print(f"[AUTOTUNER] {model_id} -> Latency: {avg_latency_ms:.2f}ms | Provider: {sess.get_providers()[0]}")

        return results
