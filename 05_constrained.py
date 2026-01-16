"""
05_constrained.py - Constrained decoding for guaranteed valid JSON
"""

import json
from pathlib import Path
from enum import Enum
from typing import Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel

MODEL_PATH = "./finetuned_model"

TEST_PROMPTS = [
    "Send an email to bob@gmail.com about the project",
    "What's the weather in Paris?",
    "Search for machine learning tutorials",
    "Email john@newdomain.io regarding the deadline",
    "Check weather in San Francisco",
]


# Pydantic models for outlines
class ToolName(str, Enum):
    send_email = "send_email"
    get_weather = "get_weather"
    search_web = "search_web"

class EmailArgs(BaseModel):
    to: str
    subject: str

class WeatherArgs(BaseModel):
    city: str

class SearchArgs(BaseModel):
    query: str

class SendEmailCall(BaseModel):
    tool: ToolName
    args: EmailArgs

class GetWeatherCall(BaseModel):
    tool: ToolName
    args: WeatherArgs

class SearchWebCall(BaseModel):
    tool: ToolName
    args: SearchArgs

ToolCall = Union[SendEmailCall, GetWeatherCall, SearchWebCall]


def demo_outlines():
    """Use outlines library for constrained generation."""
    print("\n" + "=" * 60)
    print("APPROACH 1: Using outlines library")
    print("=" * 60)

    try:
        import outlines
    except ImportError:
        print("outlines not installed. Run: pip install outlines")
        return

    print("\nLoading model...")
    model = outlines.models.transformers(MODEL_PATH)
    generator = outlines.generate.json(model, ToolCall)

    print("Testing...\n")
    for prompt in TEST_PROMPTS:
        full_prompt = f"User request: {prompt}\nTool call JSON: "
        result = generator(full_prompt)
        print(f"INPUT:  {prompt}")
        print(f"OUTPUT: {result}")
        print(f"VALID:  Always (constrained)")
        print("-" * 50)


class SchemaConstrainedDecoder:
    """
    Forces output to match our schema by controlling generation at each step.
    We force the JSON structure but let the model choose the values.
    """

    TOOLS = ["send_email", "get_weather", "search_web"]
    TOOL_ARGS = {
        "send_email": ["to", "subject"],
        "get_weather": ["city"],
        "search_web": ["query"],
    }

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size

    def get_token_id(self, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        return tokens[0] if tokens else None

    def force_text(self, input_ids, text):
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        tokens_tensor = torch.tensor([tokens])
        return torch.cat([input_ids, tokens_tensor], dim=-1)

    def generate_string_value(self, input_ids, max_tokens=20):
        generated = ""
        for _ in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
                logits = outputs.logits[0, -1, :]

            next_token = torch.argmax(logits).unsqueeze(0)
            decoded = self.tokenizer.decode(next_token)

            if '"' in decoded:
                generated += decoded.split('"')[0]
                break

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            generated += decoded
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        return input_ids, generated

    def generate(self, prompt):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # Force start of JSON and "tool" key
        input_ids = self.force_text(input_ids, '{"tool": "')

        # Let model pick tool (constrained to valid options)
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[0, -1, :]

        tool_scores = {}
        for tool in self.TOOLS:
            tok = self.get_token_id(tool)
            if tok is not None:
                tool_scores[tool] = logits[tok].item()

        best_tool = max(tool_scores, key=tool_scores.get)

        # Force rest of structure
        input_ids = self.force_text(input_ids, f'{best_tool}", "args": {{')

        # Generate each arg value
        args = self.TOOL_ARGS[best_tool]
        arg_values = {}

        for i, arg in enumerate(args):
            input_ids = self.force_text(input_ids, f'"{arg}": "')
            input_ids, value = self.generate_string_value(input_ids)
            arg_values[arg] = value

            if i < len(args) - 1:
                input_ids = self.force_text(input_ids, '", ')
            else:
                input_ids = self.force_text(input_ids, '"')

        # Build result
        result = f'{{"tool": "{best_tool}", "args": {{'
        for i, (arg, val) in enumerate(arg_values.items()):
            result += f'"{arg}": "{val}"'
            if i < len(arg_values) - 1:
                result += ", "
        result += "}}"

        return result


def demo_from_scratch():
    """Show constrained decoding without external libraries."""
    print("\n" + "=" * 60)
    print("APPROACH 2: From-scratch constrained decoder")
    print("=" * 60)

    print("""
How it works:
1. Force the JSON structure: {"tool": "...", "args": {...}}
2. Let the model choose from valid tool names
3. Force the correct arg keys for that tool
4. Let model generate arg values (stop at closing quote)

The structure is guaranteed, only the values are model-generated.
""")

    if not Path(MODEL_PATH).exists():
        print(f"Model not found at {MODEL_PATH}. Run 03_finetune.py first.")
        return

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    decoder = SchemaConstrainedDecoder(model, tokenizer)

    print("\nTesting...\n")
    valid_count = 0
    for prompt in TEST_PROMPTS:
        full_prompt = f"User request: {prompt}\nTool call JSON: "
        result = decoder.generate(full_prompt)

        print(f"INPUT:  {prompt}")
        print(f"OUTPUT: {result}")
        try:
            parsed = json.loads(result)
            print(f"VALID:  Yes - {parsed['tool']}({parsed['args']})")
            valid_count += 1
        except json.JSONDecodeError as e:
            print(f"VALID:  No - {e}")
        print("-" * 50)

    print(f"\nResults: {valid_count}/{len(TEST_PROMPTS)} valid")


def main():
    print("=" * 60)
    print("CONSTRAINED DECODING")
    print("=" * 60)

    print("""
The idea: at each step, mask out invalid tokens before sampling.
If we're expecting a closing brace, only } gets nonzero probability.
The model literally cannot output invalid JSON.

This is what OpenAI calls "Structured Outputs".
""")

    demo_outlines()
    demo_from_scratch()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
Fine-tuning: model learns to WANT to output correct JSON
Constrained decoding: makes it IMPOSSIBLE to output invalid JSON

Together = reliable tool use.
""")


if __name__ == "__main__":
    main()
