#!/usr/bin/env bash
set -euo pipefail

# Color Codes
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}[1/5] Executing Pre-flight Deployment Checks...${NC}"

# Check for Docker and NVIDIA Container Toolkit
if ! command -v docker &> /dev/null; then
    echo -e "${RED}ERROR: docker CLI is not installed.${NC}"
    exit 1
fi

if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}ERROR: NVIDIA GPU drivers / Container Toolkit not detected.${NC}"
    exit 1
fi

echo -e "${GREEN}[2/5] Validating Required Production Secrets...${NC}"

mkdir -p ./secrets

if [ ! -f ./secrets/ngc_api_key.txt ]; then
    echo -e "${RED}ERROR: ./secrets/ngc_api_key.txt is missing.${NC}"
    exit 1
fi

if [ ! -f ./secrets/milvus_token.txt ]; then
    echo -e "${RED}ERROR: ./secrets/milvus_token.txt is missing.${NC}"
    exit 1
fi

chmod 600 ./secrets/*.txt

echo -e "${GREEN}[3/5] Building Sentinel Core Container Image...${NC}"
docker compose -f docker-compose.prod.yml build sentinel-app-core

echo -e "${GREEN}[4/5] Launching NIM Microservices & Core Cluster...${NC}"
docker compose -f docker-compose.prod.yml up -d

echo -e "${GREEN}[5/5] Verifying Deployment Health...${NC}"
sleep 10
docker compose -f docker-compose.prod.yml ps

echo -e "${GREEN}=== Deployment Complete ===${NC}"
