fn main() {
    println!("cargo:rerun-if-changed=src/lib.rs");
    
    // Link CUDA runtime libraries
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
    println!("cargo:rustc-link-lib=cudart");
    
    // Link TensorRT execution libraries
    println!("cargo:rustc-link-lib=nvinfer");
    println!("cargo:rustc-link-lib=onnxruntime");
}
