from mlc_llm import MLCEngine

# cmake .. && cmake --build . --parallel $(nproc) 

# Create engine
model = "/home/cjmcv/project/llm_models/Qwen/Qwen2-1___5B-Instruct-q4f16_1-MLC"
engine = MLCEngine(model=model,
                   model_lib="/home/cjmcv/project/llm_models/Qwen/libs/Qwen2-1___5B-Instruct-q4f16_1-MLC-cuda.so")

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
