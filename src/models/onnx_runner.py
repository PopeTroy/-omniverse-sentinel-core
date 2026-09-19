import asyncio
import numpy as np
import onnxruntime as ort
from typing import Dict, Any, List

class ParallelONNXEngine:
    """Executes thousands of parallel ONNX vision, anomaly, and feature models

    using NVIDIA TensorRT execution providers.
    """
    def __init__(self, model_paths: List[str]):
        self.sessions: List[ort.InferenceSession] = []
        providers = [
            ('TensorRTExecutionProvider', {
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': './trt_cache'
            }),
            'CUDAExecutionProvider',
            'CPUExecutionProvider'
        ]
        
        for path in model_paths:
            session = ort.InferenceSession(path, providers=providers)
            self.sessions.append(session)

    async def execute_model(self, session: ort.InferenceSession, input_data: np.ndarray) -> np.ndarray:
        input_name = session.get_inputs()[0].name
        loop = asyncio.get_running_loop()
        # Non-blocking executor call for heavy GPU compute
        result = await loop.run_in_executor(
            None, lambda: session.run(None, {input_name: input_data})
        )
        return result[0]

    async def process_parallel_streams(self, data_batch: List[np.ndarray]) -> List[np.ndarray]:
        tasks = [
            self.execute_model(session, data) 
            for session, data in zip(self.sessions, data_batch)
        ]
        return await asyncio.gather(*tasks)
