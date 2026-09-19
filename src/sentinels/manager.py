import asyncio
import logging
from typing import Dict, List
from src.sentinels.watchdog import IndustryPillarWatchdog
from src.nim.service_client import NVIDIA_NIM_Client
from src.core.hyper_matrix import IntelligenceCoreHyperMatrix

logger = logging.getLogger("SentinelManager")

class EphemeralSentinelManager:
    """Manages the spawning, routing, and destruction of ephemeral watchdog sentinels."""

    def __init__(self, nim_client: NVIDIA_NIM_Client, core: IntelligenceCoreHyperMatrix):
        self.nim_client = nim_client
        self.core = core
        self.active_sentinels: Dict[str, IndustryPillarWatchdog] = {}

    async def spawn_sentinel(self, pillar_id: str) -> IndustryPillarWatchdog:
        """Instantiate an ephemeral watchdog for a specific task or pillar."""
        logger.info(f"[SPAWN] Instantiating Ephemeral Sentinel for: {pillar_id}")
        sentinel = IndustryPillarWatchdog(pillar_id, self.nim_client, self.core)
        self.active_sentinels[pillar_id] = sentinel
        return sentinel

    async def terminate_sentinel(self, pillar_id: str):
        """Reclaim resources immediately upon task completion."""
        if pillar_id in self.active_sentinels:
            del self.active_sentinels[pillar_id]
            logger.info(f"[TERMINATE] Ephemeral Sentinel dissolved: {pillar_id}")

    async def dispatch_event(self, pillar_id: str, stream_data, knowledge_db: list):
        sentinel = await self.spawn_sentinel(pillar_id)
        try:
            await sentinel.monitor_event_stream(stream_data, knowledge_db)
        finally:
            await self.terminate_sentinel(pillar_id)
