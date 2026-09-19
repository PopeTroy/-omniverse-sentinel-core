import asyncio
import numpy as np
from src.models.onnx_runner import ParallelONNXEngine
from src.nim.service_client import NVIDIA_NIM_Client
from src.rag.ephemeral_pipeline import EphemeralRAGPipeline
from src.core.hyper_matrix import IntelligenceCoreHyperMatrix

class IndustryPillarWatchdog:
    """Ephemeral watchdog node monitoring specific industry pillars (e.g. Finance, Healthcare)."""
    
    def __init__(self, pillar_id: str, nim_client: NVIDIA_NIM_Client, core: IntelligenceCoreHyperMatrix):
        self.pillar_id = pillar_id
        self.nim_client = nim_client
        self.core = core

    async def monitor_event_stream(self, stream_data: np.ndarray, knowledge_db: list):
        # Step 1: Ephemeral RAG Pipeline Instantiation
        rag = EphemeralRAGPipeline(self.nim_client)
        
        try:
            # Step 2: Trigger conditions analysis
            data_summary = f"Pillar [{self.pillar_id}] metric mean: {np.mean(stream_data):.4f}"
            retrieved_context = await rag.spawn_and_retrieve(data_summary, knowledge_db)
            
            # Step 3: NVIDIA NIM Reasoning Cycle
            reasoning_response = await self.nim_client.generate_reasoning(
                prompt=f"Analyze anomaly state for {data_summary}",
                context=retrieved_context
            )
            
            # Step 4: Extract updates & synthesize into Central Intelligence Core
            insight_text = reasoning_response["choices"][0]["message"]["content"]
            delta_vector = np.array(await self.nim_client.generate_embeddings(insight_text))
            
            self.core.assimilate_gradient_delta(delta_vector)
            print(f"[{self.pillar_id}] Core State Assimilated. Current Core Entropy: {self.core.get_core_entropy():.4f}")

        finally:
            # Cleanup Ephemeral state immediately
            await rag.dissolve()
