# Tool Use From Scratch

How do OpenAI and Anthropic get LLMs to reliably output JSON for tool use?

Two techniques:

1. **Fine-tuning**: Train the model on examples of correct tool calls
2. **Constrained decoding**: At inference time, only allow valid tokens

This repo demonstrates both with a tiny model you can run on your laptop.

## Prerequisites

You need Python 3.10 or higher. Check your version:

```bash
python3 --version
```

If you don't have Python, download it from [python.org](https://www.python.org/downloads/) or use your package manager:

```bash
# macOS (with Homebrew)
brew install python

# Ubuntu/Debian
sudo apt install python3 python3-venv

# Windows - download from python.org
```

## Setup

First, clone or download this repo and cd into it:

```bash
cd tool-use-from-scratch
```

Create a virtual environment. This keeps the dependencies isolated from your system Python:

```bash
python3 -m venv .venv
```

Activate the virtual environment:

```bash
# macOS/Linux
source .venv/bin/activate

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

)` at the start of your terminal prompt now.

Install the dependencies:

```bash
pip install -r requirements.txt
```

This will download PyTorch, Hugging Face transformers, and other packages. It might take a few minutes.

## Running the Demo

Run the scripts in order:

```bash
# 1. See that base model can't output tool calls
python 01_baseline.py

# 2. Generate training data
python 02_create_dataset.py

# 3. Fine-tune the model (takes a few minutes on CPU)
python 03_finetune.py

# 4. Test the fine-tuned model
python 04_inference.py

# 5. Add constrained decoding for guaranteed valid output
python 05_constrained.py
```

## What's Going On

### The Problem

Base language models are trained on internet text. They've seen JSON, but they're not great at producing valid JSON reliably. They might:

- Forget a comma or closing brace
- Add commentary before or after the JSON
- Hallucinate field names
- Mix up the structure

Run `01_baseline.py` to see this:

```
INPUT:  Send an email to jack@gmail.com about the meeting
OUTPUT: I'm happy to help you send an email! Here's what I would...
VALID JSON: No
```

### Solution 1: Fine-tuning

We fine-tune the model on hundreds of examples of correct tool calls:

```json
{"prompt": "Send an email to jack@gmail.com about the meeting",
 "completion": "{\"tool\": \"send_email\", \"args\": {\"to\": \"jack@gmail.com\", \"subject\": \"the meeting\"}}"}
```

After fine-tuning (`03_finetune.py`), the model learns the pattern. Test it with `04_inference.py`:

```
INPUT:  Send an email to bob@gmail.com about the project
OUTPUT: {"tool": "send_email", "args": {"to": "bob@gmail.com", "subject": "the project"}}
VALID:  Yes
```

Success rate goes from ~0% to 40-60%. Better, but not perfect. The model learned the pattern probabilistically - it usually gets it right, but still makes mistakes.

### Solution 2: Constrained Decoding

At each generation step, an LLM produces probabilities for every token in its vocabulary. Normally you sample from this distribution.

With constrained decoding, you mask out invalid tokens before sampling. If we've generated `{"tool": "send_email", "args": {` and the schema says the next field must be `"to"`, then every other token gets probability zero.

The model **literally cannot** output invalid JSON.

Run `05_constrained.py`:

```
INPUT:  Send an email to bob@gmail.com about the project
OUTPUT: {"tool": "send_email", "args": {"to": "bob@gmail.com", "subject": "the project"}}
VALID:  Always (constrained by schema)
```

100% valid, every time.

## The Schema

We use three simple tools:

```python
TOOLS = {
    "send_email": {"args": ["to", "subject"]},
    "search_web": {"args": ["query"]},
    "get_weather": {"args": ["city"]},
}
```

Output format:
```json
{"tool": "send_email", "args": {"to": "alice@example.com", "subject": "meeting"}}
{"tool": "get_weather", "args": {"city": "London"}}
{"tool": "search_web", "args": {"query": "python tutorials"}}
```

## Why Both Techniques?

| Technique | What it does | Limitation |
|-----------|--------------|------------|
| Fine-tuning | Makes model *want* to output correct JSON | Probabilistic, can still make errors |
| Constrained decoding | Makes it *impossible* to output invalid JSON | Slower, can produce weird valid JSON |

## What the Big Labs Do

This repo demonstrates the same principles OpenAI and Anthropic use:

| This repo | Production systems |
|-----------|-------------------|
| 500 training examples | Millions of examples |
| distilgpt2 (82M params) | GPT-5, Claude (100B+ params) |
| 3 tools | Hundreds of tools |
| Simple constraints | Sophisticated grammar engines |

The fundamentals are identical:
1. **SFT (Supervised Fine-Tuning)**: Train on examples of correct tool calls
2. **Constrained decoding**: Mechanically enforce valid outputs
3. **RLHF** (not shown here): Reinforce good tool-use behavior through human feedback

## Questions?

Come and ask me!