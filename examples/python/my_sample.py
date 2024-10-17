from mlc_llm import MLCEngine

# export PYTHONPATH=/root/llm/mlc-llm/3rdparty/tvm/python:$PYTHONPATH
# export MLC_LLM_SOURCE_DIR=/root/llm/mlc-llm 
# export PYTHONPATH=$MLC_LLM_SOURCE_DIR/python:$PYTHONPATH
# alias mlc_llm="python -m mlc_llm"
# conda activate tvm-build-venv

# cmake .. && cmake --build . --parallel $(nproc) 

# mlc_llm convert_weight /root/autodl-tmp/Qwen/Qwen2-7B-Instruct  --quantization q4f16_1  -o /root/autodl-tmp/Qwen/Qwen2-7B-Instruct-q4f16_1-MLC
# mlc_llm gen_config /root/autodl-tmp/Qwen/Qwen2-7B-Instruct  --quantization q4f16_1 --conv-template qwen2 -o /root/autodl-tmp/Qwen/Qwen2-7B-Instruct-q4f16_1-MLC
# mlc_llm compile /root/autodl-tmp/Qwen/Qwen2-7B-Instruct-q4f16_1-MLC/mlc-chat-config.json --device cuda -o /root/autodl-tmp/Qwen/libs/Qwen2-7B-Instruct-q4f16_1-MLC-cuda.so
# python examples/python/my_sample.py

# Create engine
model = "/root/autodl-tmp/Qwen/Qwen2-7B-Instruct-q4f16_1-MLC"
engine = MLCEngine(model=model,
                   model_lib="/root/autodl-tmp/Qwen/libs/Qwen2-7B-Instruct-q4f16_1-MLC-cuda.so")

# Run chat completion in OpenAI API.
for response in engine.chat.completions.create(
    messages=[{"role": "user", "content": "What is the meaning of life?"}],
    model=model,
    stream=True,
):
    for choice in response.choices:
        print(choice.delta.content, end="", flush=True)
print("\n")

print(engine.metrics())
engine.terminate()
