import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# model_name = "Qwen/Qwen3-4B"
# model_name = "Qwen/Qwen3-1.7B"
model_name = "Qwen/Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    torch_dtype=torch.float16
)

def review_paper(paper_text):
    paper_tokens = tokenizer(
        paper_text,
        truncation=True,
        max_length=7500,
        return_tensors="pt"
    )

    paper_text = tokenizer.decode(
        paper_tokens["input_ids"][0],
        skip_special_tokens=True
    )

    prompt = f"""
/no_think

You are an academic peer reviewer.

Write ONLY the final peer review in English.
Do not explain your reasoning.
Do not describe what you are going to do.
Do not ask the user questions.

Use exactly these sections:

1. Summary
2. Strengths
3. Weaknesses
4. Questions
5. Suggested Improvements
6. Final Score (1-10)

PAPER:
{paper_text}

Now output the final review only.
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    print("Input tokens:", inputs["input_ids"].shape[1])

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.1
        )

    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )