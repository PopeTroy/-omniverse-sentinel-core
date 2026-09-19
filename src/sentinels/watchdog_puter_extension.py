# src/sentinels/watchdog_puter_extension.py
import asyncio
from src.rag.puter_fs_manager import PuterFSGPUManager

async def run_ephemeral_watchdog_with_puter_fs(pillar_id: str, stream_payload: bytes):
    puter_fs = PuterFSGPUManager()

    # 1. Evaluate load and dynamically auto-scale GPU RAG instances as needed
    workload_intensity = len(stream_payload) / 1024.0
    active_gpus = await puter_fs.allocate_gpu_instances_as_needed(load_metric=workload_intensity)

    # 2. Setup Ephemeral Directory Tree via Puter.fs
    working_dir = f"/sentinels/{pillar_id}"
    await puter_fs.mkdir(working_dir)

    # 3. Write incoming stream state to Puter.fs
    input_file = f"{working_dir}/stream_input.bin"
    await puter_fs.write(input_file, stream_payload)

    # 4. Stat the file to verify size
    file_stat = await puter_fs.stat(input_file)
    print(f"[{pillar_id}] Input Stat: {file_stat}")

    # 5. Copy/Paste state artifact to GPU scratchpad
    scratch_dir = f"/sys/gpu_instances/ephemeral-gpu-rag-instance-1"
    await puter_fs.copy(input_file, f"{scratch_dir}/processing_chunk.bin")

    # 6. Process complete -> Clean up Puter.fs paths immediately
    await puter_fs.delete(input_file)
    await puter_fs.delete(working_dir)
