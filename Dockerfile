FROM nvcr.io/nvidia/pytorch:24.01-py3

# Set system env
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies & Rust toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    cmake \
    git \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source repository
COPY . .

# Compile Rust Native Module
RUN cd src/native && cargo build --release

# Expose Sentinel Core Service Ports
EXPOSE 8080 9090

CMD ["python3", "main.py"]
