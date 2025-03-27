#!/bin/bash
set -x

# Check if a model parameter was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <model> <lora_count>"
    echo "Available models:"
    echo "  OpenELM-1.1B     - OpenELM 1.1B model"
    echo "  Llama3.2-3B      - Llama 3.2 3B model"
    echo "  Llama3.1-8B      - Llama 3.1 8B model"
    exit 1
fi

MODEL=$1
LORA_COUNT=$2

# Default adapter cache sizes
ADAPTER_CACHE_OPENELM=50
ADAPTER_CACHE_LLAMA32=50
ADAPTER_CACHE_LLAMA31=50

case $MODEL in
    OpenELM-1.1B)
        # Set adapter_cache_size to LORA_COUNT if LORA_COUNT is smaller
        ADAPTER_CACHE=$ADAPTER_CACHE_OPENELM
        if [ -n "$LORA_COUNT" ] && [ "$LORA_COUNT" -lt "$ADAPTER_CACHE" ]; then
            ADAPTER_CACHE=$LORA_COUNT
        fi
        
        echo "Starting OpenELM-1.1B server with $LORA_COUNT LoRAs (adapter cache: $ADAPTER_CACHE)..."
        ./llama-server -m ./models/OpenELM-1.1B/Q4_0-00001-of-00001.gguf -c 25600 --lora_repeated ./models/aac/lora.gguf $LORA_COUNT --adapter_cache_size $ADAPTER_CACHE --parallel 60 --batch_lora true --n-gpu-layers 999
        ;;
    Llama3.2-3B)
        # Set adapter_cache_size to LORA_COUNT if LORA_COUNT is smaller
        ADAPTER_CACHE=$ADAPTER_CACHE_LLAMA32
        if [ -n "$LORA_COUNT" ] && [ "$LORA_COUNT" -lt "$ADAPTER_CACHE" ]; then
            ADAPTER_CACHE=$LORA_COUNT
        fi
        
        echo "Starting Llama3.2-3B server with $LORA_COUNT LoRAs (adapter cache: $ADAPTER_CACHE)..."
        ./llama-server -m ./models/Llama3.2-3B/Llama-Sentient-3.2-3B-Instruct.Q4_0.gguf -c 51200 --lora_repeated ./models/Llama3.2-3B/Meta-Llama-3.2-3B-MEDAL-finetune-F16-LoRA.gguf $LORA_COUNT --adapter_cache_size $ADAPTER_CACHE --parallel 40 --batch_lora true --n-gpu-layers 999
        ;;
    Llama3.1-8B)
        # Set adapter_cache_size to LORA_COUNT if LORA_COUNT is smaller
        ADAPTER_CACHE=$ADAPTER_CACHE_LLAMA31
        if [ -n "$LORA_COUNT" ] && [ "$LORA_COUNT" -lt "$ADAPTER_CACHE" ]; then
            ADAPTER_CACHE=$LORA_COUNT
        fi
        
        echo "Starting Llama3.1-8B server with $LORA_COUNT LoRAs (adapter cache: $ADAPTER_CACHE)..."
        ./llama-server -m ./models/Llama3.1-8B/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf -c 102400 --lora_repeated ./models/Llama3.1-8B/Llama-8B-3_1-Instruct-orca-ORPO-F16-LoRA.gguf $LORA_COUNT --adapter_cache_size $ADAPTER_CACHE --parallel 20 --batch_lora true --n-gpu-layers 999
        ;;
    *)
        echo "Invalid model selection: $MODEL"
        echo "Usage: $0 <model> [lora_count]"
        echo "Available models:"
        echo "  OpenELM-1.1B     - OpenELM 1.1B model"
        echo "  Llama3.2-3B      - Llama 3.2 3B model"
        echo "  Llama3.1-8B      - Llama 3.1 8B model"
        exit 1
        ;;
esac