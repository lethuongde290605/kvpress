# Demo method score trả về tensor
import torch
import sys
sys.path.append('/home/imdeeslt/Study/[2025] NCKH/Task/Week4/kvpress')

from kvpress.presses.knorm_press import KnormPress
from kvpress.presses.random_press import RandomPress
from kvpress.pipeline import KVPressTextGenerationPipeline

from transformers import pipeline

print("=== DEMO METHOD SCORE ===")
device = "cuda:0"
# Đổi sang model open-source:
model = "mistralai/Mistral-7B-Instruct-v0.2"
model_kwargs = {"attn_implementation": "eager"}

pipe = pipeline("kv-press-text-generation", model=model, device=device, model_kwargs=model_kwargs)

context = "A very long text you want to compress once and for all"
question = "\nA question about the compressed context"  # optional

press = KnormPress(compression_ratio=0.5)
answer = pipe(context, question=question, press=press)["answer"]

print(f'question: {question}')
print(f'answer: {answer}')
