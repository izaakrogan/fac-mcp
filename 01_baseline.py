"""
01_baseline.py - Show that base distilgpt2 can't output tool calls
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading distilgpt2...")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token

DESIRED_FORMAT = '''
We want the model to output valid JSON like:
{"tool": "send_email", "args": {"to": "jack@gmail.com", "subject": "meeting"}}
{"tool": "get_weather", "args": {"city": "London"}}
{"tool": "search_web", "args": {"query": "python tutorials"}}
'''

TEST_PROMPTS = [
    "Send an email to jack@gmail.com about the meeting",
    "What's the weather in London?",
    "Search the web for python tutorials",
    "Email sarah@company.com with subject quarterly report",
    "Look up the weather in Tokyo",
]

def genrate_response(prompt: str, max_new_tokens: int = 50) -> str:
    full_prompt = f"User request: {prompt}\nTool call JSON: "
    inputs = tokenizer(full_prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated = full_output[len(full_prompt):]
    return generated.strip()


def try_parse_json(text: str) -> bool:
    import json
    try:
        if "{" in text:
            start = text.index("{")
            end = text.rfind("}") + 1
            if end > start:
                json.loads(text[start:end])
                return True
        return False
    except (json.JSONDecodeError, ValueError):
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("BASELINE TEST: Can distilgpt2 output valid tool-call JSON?")
    print("=" * 60)
    print(DESIRED_FORMAT)
    print("=" * 60)
    print("\nTesting base model on tool-call requests...\n")

    valid_count = 0
    for prompt in TEST_PROMPTS:
        print(f"INPUT:  {prompt}")
        output = genrate_response(prompt)
        is_valid = try_parse_json(output)
        valid_count += is_valid

        display_output = output[:100] + "..." if len(output) > 100 else output
        print(f"OUTPUT: {display_output}")
        print(f"VALID JSON: {'Yes' if is_valid else 'No'}")
        print("-" * 40)

    print(f"\nResults: {valid_count}/{len(TEST_PROMPTS)} outputs were valid JSON")
    print("\nThe base model has no idea what we want.")
    print("We need to fine-tune it on examples of correct tool calls.")
