"""
04_inference.py - Test the fine-tuned model
"""

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "./finetuned_model"

TESET_PROMPTS = [
    "Send an email to bob@gmail.com about the project",
    "What's the weather in Paris?",
    "Search for machine learning tutorials",
    "Email john@newdomain.io regarding the deadline",
    "Check weather in San Francisco",
    "Look up how to cook pasta",
    "Send alice@test.com an email about tomorrow's meeting",
    "What's the temperature in Moscow?",
    "Find videos about python programming",
    "Can you email support@company.com about a bug?",
    "I need the weather for Chicago",
    "Search the internet for best laptops 2024",
]


def load_model():
    if not Path(MODEL_PATH).exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Run 03_finetune.py first.")
        return None, None

    print(f"Loading model from {MODEL_PATH}...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    return model, tokenizer


def generate_tool_call(model, tokenizer, prompt, max_new_tokens=60):
    full_prompt = f"User request: {prompt}\nTool call JSON: "
    inputs = tokenizer(full_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = full_output[len(full_prompt):]
    return generated.split("\n")[0].strip()


def extract_json(text):
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text[start:], start):
        if escape_next:
            escape_next = False
            continue
        if char == "\\":
            escape_next = True
            continue
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:i+1]
    return None


def validate_tool_call(text):
    SCHEMA = {
        "send_email": ["to", "subject"],
        "get_weather": ["city"],
        "search_web": ["query"],
    }

    json_str = extract_json(text)
    if json_str is None:
        return False, "No JSON found", None

    try:
        data = json.loads(json_str)
        if "tool" not in data:
            return False, "Missing 'tool' field", json_str
        if "args" not in data:
            return False, "Missing 'args' field", json_str

        tool = data["tool"]
        if tool not in SCHEMA:
            return False, f"Unknown tool: {tool}", json_str

        for arg in SCHEMA[tool]:
            if arg not in data["args"]:
                return False, f"Missing arg: {arg}", json_str

        return True, "Valid", json_str
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {e}", json_str


def main():
    print("=" * 60)
    print("INFERENCE TEST: How well did fine-tuning work?")
    print("=" * 60)

    model, tokenizer = load_model()
    if model is None:
        return

    print(f"\nTesting on {len(TESET_PROMPTS)} prompts...\n")

    results = []
    for prompt in TESET_PROMPTS:
        output = generate_tool_call(model, tokenizer, prompt)
        is_valid, error, extracted = validate_tool_call(output)
        results.append((prompt, output, is_valid, error, extracted))

        print(f"INPUT:  {prompt}")
        print(f"RAW:    {output[:80]}{'...' if len(output) > 80 else ''}")
        if extracted:
            print(f"JSON:   {extracted}")
        print(f"VALID:  {'Yes' if is_valid else f'No - {error}'}")
        print("-" * 50)

    valid_count = sum(1 for r in results if r[2])
    success_rate = valid_count / len(results) * 100

    print("\n" + "=" * 60)
    print(f"Valid outputs: {valid_count}/{len(results)} ({success_rate:.0f}%)")
    print("=" * 60)

    print("""
Fine-tuning helped! The model learnt the pattern but still makes mistakes.
It's probabilistic - it can still output invalid JSON sometimes.

For guaranteed valid output, run 05_constrained.py
""")


if __name__ == "__main__":
    main()
