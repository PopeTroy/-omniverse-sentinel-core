import logging
import numpy as np
from typing import Dict, Any, List, Optional
from src.core.divine_tactics import ShinobiChakraResonator, OcularAbility

logger = logging.getLogger("ShinobiPuterFS")

class PuterFSGPUManager:
    """Virtual Filesystem & Ephemeral GPU Pool powered by Puter.fs and Shinobi Ocular Abilities."""

    def __init__(self):
        self.fs_store: Dict[str, bytes] = {}
        self.active_gpu_instances: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # Puter.fs Operations with Sharingan / Rinnegan Optimization
    # -------------------------------------------------------------------------
    async def write(self, path: str, data: bytes) -> bool:
        """Write binary state or embeddings to the Puter.fs virtual directory."""
        self.fs_store[path] = data
        logger.info(f"[Puter.fs WRITE] [{OcularAbility.SHARINGAN_PREDICTION.value}] Path: {path} ({len(data)} bytes)")
        return True

    async def read(self, path: str) -> Optional[bytes]:
        """Read state artifacts from Puter.fs using zero-copy extraction."""
        data = self.fs_store.get(path)
        if data is None:
            logger.error(f"[Puter.fs READ] Path not found: {path}")
            return None
        logger.info(f"[Puter.fs READ] [{OcularAbility.RINNEGAN_SURA_PATH.value}] Extracted {len(data)} bytes from {path}")
        return data

    async def copy(self, src_path: str, dest_path: str) -> bool:
        """Copy state within Puter.fs directory tree."""
        data = await self.read(src_path)
        if data is not None:
            return await self.write(dest_path, data)
        return False

    async def paste(self, src_path: str, dest_dir: str) -> bool:
        """Paste state artifact into target virtual directory."""
        filename = src_path.split("/")[-1]
        dest_path = f"{dest_dir.rstrip('/')}/{filename}"
        return await self.copy(src_path, dest_path)

    async def delete(self, path: str) -> bool:
        """Delete path and instantly reclaim virtual storage."""
        if path in self.fs_store:
            del self.fs_store[path]
            logger.info(f"[Puter.fs DELETE] Reclaimed path: {path}")
            return True
        return False

    async def stat(self, path: str) -> Dict[str, Any]:
        """Inspect file metadata and size."""
        data = self.fs_store.get(path)
        if data is None:
            return {"exists": False, "size": 0}
        return {
            "exists": True,
            "size": len(data),
            "path": path,
            "type": "file" if not path.endswith("/") else "directory"
        }

    async def mkdir(self, dir_path: str) -> bool:
        """Create virtual directory path in Puter.fs."""
        formatted_dir = dir_path if dir_path.endswith("/") else f"{dir_path}/"
        self.fs_store[formatted_dir] = b""
        logger.info(f"[Puter.fs MKDIR] Directory initialized: {formatted_dir}")
        return True

    # -------------------------------------------------------------------------
    # Tailed Beasts Chakra GPU Auto-Scaling Engine
    # -------------------------------------------------------------------------
    async def allocate_gpu_instances_with_chakra(self, payload_bytes: int) -> int:
        """Scales GPU RAG instances dynamically based on Kurama/Gyūki Chakra Reserves."""
        chakra_multiplier = ShinobiChakraResonator.calculate_chakra_surge(payload_bytes)
        target_instances = int(np.ceil(chakra_multiplier * 2))  # Scale factor
        current_instances = len(self.active_gpu_instances)

        if target_instances > current_instances:
            to_spawn = target_instances - current_instances
            for i in range(to_spawn):
                instance_id = f"chakra-gpu-rag-{current_instances + i + 1}"
                self.active_gpu_instances.append({
                    "id": instance_id,
                    "status": "CHAKRA_MODE_ACTIVE",
                    "allocated_vram_mb": 16384
                })
                await self.mkdir(f"/sys/gpu_instances/{instance_id}")
            logger.info(f"[TAILED BEAST AUTO-SCALE] Infusing Chakra ({chakra_multiplier}x multiplier). Spawned {to_spawn} GPU workers. Total: {len(self.active_gpu_instances)}")

        elif target_instances < current_instances:
            to_remove = current_instances - target_instances
            for _ in range(to_remove):
                removed = self.active_gpu_instances.pop()
                await self.delete(f"/sys/gpu_instances/{removed['id']}/")
            logger.info(f"[TAILED BEAST AUTO-SCALE] Reclaiming Chakra. Dissolved {to_remove} GPU workers. Total: {len(self.active_gpu_instances)}")

        return len(self.active_gpu_instances)
