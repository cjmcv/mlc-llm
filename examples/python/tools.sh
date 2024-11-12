#!/bin/sh

# 01 for autodl
# usage: source examples/python/tools.sh 0

# tvm: cp ../cmake/config.cmake .
# mlc: python ../cmake/gen_cmake_config.py
#
# cmake .. && cmake --build . --parallel $(nproc) 


setup_on_autodl() {
    echo "setup_on_autodl"
    export PYTHONPATH=/root/llm/mlc-llm/3rdparty/tvm/python:$PYTHONPATH
    export MLC_LLM_SOURCE_DIR=/root/llm/mlc-llm 
    export PYTHONPATH=$MLC_LLM_SOURCE_DIR/python:$PYTHONPATH
    alias mlc_llm="python -m mlc_llm"

    conda init
    conda activate tvm-build-venv    
}

cvt_on_autodl() {
    echo "cvt_on_autodl"
    rm /root/autodl-tmp/Qwen/Qwen2-1___5B-Instruct-q4f16_1-MLC /root/autodl-tmp/Qwen/Qwen2-1___5B-Instruct-q4f16_1-MLC /root/autodl-tmp/Qwen/libs/Qwen2-1___5B-Instruct-q4f16_1-MLC-cuda.so -r
    mkdir /root/autodl-tmp/Qwen/libs/

    mlc_llm convert_weight /root/autodl-tmp/Qwen/Qwen2-1___5B-Instruct  --quantization q4f16_1  -o /root/autodl-tmp/Qwen/Qwen2-1___5B-Instruct-q4f16_1-MLC
    mlc_llm gen_config /root/autodl-tmp/Qwen/Qwen2-1___5B-Instruct  --quantization q4f16_1 --conv-template qwen2 -o /root/autodl-tmp/Qwen/Qwen2-1___5B-Instruct-q4f16_1-MLC
    mlc_llm compile /root/autodl-tmp/Qwen/Qwen2-1___5B-Instruct-q4f16_1-MLC/mlc-chat-config.json --device cuda -o /root/autodl-tmp/Qwen/libs/Qwen2-1___5B-Instruct-q4f16_1-MLC-cuda.so
}

setup_on_host() {
    echo "setup_on_host"
    export PYTHONPATH=/home/shared_dir/llm/mlc-llm/3rdparty/tvm/python:$PYTHONPATH
    export MLC_LLM_SOURCE_DIR=/home/shared_dir/llm/mlc-llm
    export PYTHONPATH=$MLC_LLM_SOURCE_DIR/python:$PYTHONPATH
    alias mlc_llm="python -m mlc_llm"

    conda init
    conda activate tvm-build-venv    
}

cvt_on_host() {
    echo "cvt_on_host"
    rm /home/shared_dir/llm/Qwen/Qwen2-1___5b-instruct-q4f16_1-MLC /home/shared_dir/llm/Qwen/Qwen2-1___5b-Instruct-q4f16_1-MLC /home/shared_dir/llm/Qwen/libs/Qwen2-1___5b-Instruct-q4f16_1-MLC-cuda.so -r
    mkdir /home/shared_dir/llm/Qwen/libs/

    mlc_llm convert_weight /home/shared_dir/llm/Qwen/Qwen2-1___5b-instruct --quantization q4f16_1  -o /home/shared_dir/llm/Qwen/Qwen2-1___5b-instruct-q4f16_1-MLC
    mlc_llm gen_config /home/shared_dir/llm/Qwen/Qwen2-1___5b-Instruct  --quantization q4f16_1 --conv-template qwen2 -o /home/shared_dir/llm/Qwen/Qwen2-1___5b-Instruct-q4f16_1-MLC
    mlc_llm compile /home/shared_dir/llm/Qwen/Qwen2-1___5b-Instruct-q4f16_1-MLC/mlc-chat-config.json --device cuda -o /home/shared_dir/llm/Qwen/libs/Qwen2-1___5b-Instruct-q4f16_1-MLC-cuda.so
}

if [ $1 -eq 0 ]; then
    setup_on_autodl
elif [ $1 -eq 1 ]; then
    cvt_on_autodl
elif [ $1 -eq 2 ]; then
    setup_on_host
elif [ $1 -eq 3 ]; then
    cvt_on_host
fi