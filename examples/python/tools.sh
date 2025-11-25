#!/bin/sh

# usage: source examples/python/tools.sh 0/1

# tvm: cp ../cmake/config.cmake .
# mlc: python ../cmake/gen_cmake_config.py
#
# cmake .. && cmake --build . --parallel $(nproc) 

# from modelscope import snapshot_download
# snapshot_download('Qwen/Qwen2-1.5B-Instruct', cache_dir='./')

setup_on_host() {
    echo "setup_on_host"

    conda init
    conda activate venv

    export PYTHONPATH=/home/cjmcv/project/mlc-llm/3rdparty/tvm/python:$PYTHONPATH
    export MLC_LLM_SOURCE_DIR=/home/cjmcv/project/mlc-llm
    export PYTHONPATH=$MLC_LLM_SOURCE_DIR/python:$PYTHONPATH
    alias mlc_llm="python -m mlc_llm"
}

cvt_on_host() {
    echo "cvt_on_host"
    rm /home/cjmcv/project/llm_models/Qwen/Qwen2-1___5B-Instruct-q4f16_1-MLC /home/cjmcv/project/llm_models/Qwen/Qwen2-1___5B-Instruct-q4f16_1-MLC /home/cjmcv/project/llm_models/Qwen/libs/Qwen2-1___5B-Instruct-q4f16_1-MLC-cuda.so -r
    mkdir /home/cjmcv/project/llm_models/Qwen/libs/

    mlc_llm convert_weight /home/cjmcv/project/llm_models/Qwen/Qwen2-1___5B-Instruct --quantization q4f16_1  -o /home/cjmcv/project/llm_models/Qwen/Qwen2-1___5B-Instruct-q4f16_1-MLC
    mlc_llm gen_config /home/cjmcv/project/llm_models/Qwen/Qwen2-1___5B-Instruct  --quantization q4f16_1 --conv-template qwen2 -o /home/cjmcv/project/llm_models/Qwen/Qwen2-1___5B-Instruct-q4f16_1-MLC
    mlc_llm compile /home/cjmcv/project/llm_models/Qwen/Qwen2-1___5B-Instruct-q4f16_1-MLC/mlc-chat-config.json --device cuda -o /home/cjmcv/project/llm_models/Qwen/libs/Qwen2-1___5B-Instruct-q4f16_1-MLC-cuda.so
}

if [ $1 -eq 0 ]; then
    setup_on_host
elif [ $1 -eq 1 ]; then
    cvt_on_host
fi
