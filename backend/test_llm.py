import sys
import os

# This forces Python to recognize the root folder, fixing the import error
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# If test_llm.py is already in the root folder, we just use the current directory:
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from backend.llm_engine import LLMEngine
except ModuleNotFoundError:
    from llm_engine import LLMEngine

print("Loading MLX Llama-3 model... (this will take 10-15 seconds)")
engine = LLMEngine()

prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nExplain why sleep is important in 2 sentences.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

print("\nPrompting model...\n")
print("Response: ", end="")

# Consume the stream
for token in engine.generate_response_stream(prompt, max_tokens=100):
    print(token, end="")
    sys.stdout.flush()

print("\n\nDone!")
