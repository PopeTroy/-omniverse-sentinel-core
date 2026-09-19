import ctypes
import numpy as np
import os

class RustBatchOutput(ctypes.Structure):
    _fields_ = [
        ("data_ptr", ctypes.POINTER(ctypes.c_float)),
        ("length", ctypes.c_int)
    ]

class RustNativeDispatcher:
    """Python bridge invoking native Rust thread-pools for parallel C-API ONNX execution."""
    
    def __init__(self, lib_path: str = "./src/native/target/release/librust_onnx_dispatch.so"):
        if os.path.exists(lib_path):
            self.lib = ctypes.CDLL(lib_path)
            self.lib.rust_parallel_onnx_dispatch.argtypes = [
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int
            ]
            self.lib.rust_parallel_onnx_dispatch.restype = RustBatchOutput
            self.lib.free_rust_buffer.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            self.has_native = True
        else:
            print("[RUST BINDINGS] Native shared library not found. Falling back to Pure Python ONNX execution.")
            self.has_native = False

    def dispatch_batch(self, model_paths: list[str], input_tensor: np.ndarray) -> np.ndarray:
        if not self.has_native:
            return input_tensor
            
        c_paths = (ctypes.c_char_p * len(model_paths))()
        for i, path in enumerate(model_paths):
            c_paths[i] = path.encode('utf-8')

        flattened_input = input_tensor.astype(np.float32).flatten()
        input_ptr = flattened_input.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Invoke Rust Execution Pipeline
        result = self.lib.rust_parallel_onnx_dispatch(
            c_paths,
            len(model_paths),
            input_ptr,
            len(flattened_input)
        )
        
        # Unpack C Float Array back to Numpy
        buffer_array = np.ctypeslib.as_array(result.data_ptr, shape=(result.length,))
        output_copy = np.copy(buffer_array)
        
        # Free Rust Allocations
        self.lib.free_rust_buffer(result.data_ptr, result.length)
        return output_copy
