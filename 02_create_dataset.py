"""
02_create_dataset.py - Generate synthetic training data for tool calls
"""

import json
import random
from pathlib import Path

TOOLS = {
    "send_email": {"args": ["to", "subject"]},
    "search_web": {"args": ["query"]},
    "get_weather": {"args": ["city"]},
}

NAMES = ["alice", "bob", "charlie", "diana", "eve", "frank", "grace", "henry"]
DOMAINS = ["gmail.com", "company.com", "work.org", "email.net", "outlook.com"]
CITIES = ["London", "Tokyo", "Paris", "New York", "Sydney", "Berlin", "Toronto", "Mumbai"]
SUBJECTS = [
    "the meeting", "quarterly report", "project update", "lunch plans",
    "vacation request", "bug fix", "code review", "team sync",
    "deadline reminder", "weekly summary", "feedback", "proposal"
]
SEARCH_QUERIES = [
    "python tutorials", "machine learning basics", "best restaurants nearby",
    "weather forecast", "flight prices", "hotel deals", "recipe ideas",
    "javascript frameworks", "how to learn coding", "stock prices today",
    "latest news", "movie reviews", "book recommendations"
]

EMAIL_TEMPLATES = [
    "Send an email to {email} about {subject}",
    "Email {email} with subject {subject}",
    "Write an email to {email} about {subject}",
    "Please email {email} regarding {subject}",
    "Can you send {email} an email about {subject}",
    "Message {email} about {subject}",
    "Send {email} a note about {subject}",
    "Compose an email to {email} subject {subject}",
]

WEATHER_TEMPLATES = [
    "What's the weather in {city}?",
    "Get the weather for {city}",
    "Weather in {city}",
    "How's the weather in {city}?",
    "Check the weather in {city}",
    "What's it like in {city} today?",
    "Tell me the weather in {city}",
    "Is it raining in {city}?",
]

SEARCH_TEMPLATES = [
    "Search the web for {query}",
    "Look up {query}",
    "Search for {query}",
    "Find information about {query}",
    "Google {query}",
    "Can you search {query}",
    "Look up information on {query}",
    "Search {query} online",
]


def generate_email():
    name = random.choice(NAMES)
    domain = random.choice(DOMAINS)
    return f"{name}@{domain}"


def generate_email_example():
    email = generate_email()
    subject = random.choice(SUBJECTS)
    template = random.choice(EMAIL_TEMPLATES)
    prompt = template.format(email=email, subject=subject)
    completion = json.dumps({"tool": "send_email", "args": {"to": email, "subject": subject}})
    return {"prompt": prompt, "completion": completion}


def generate_weather_example():
    city = random.choice(CITIES)
    template = random.choice(WEATHER_TEMPLATES)
    prompt = template.format(city=city)
    completion = json.dumps({"tool": "get_weather", "args": {"city": city}})
    return {"prompt": prompt, "completion": completion}


def generate_search_example():
    query = random.choice(SEARCH_QUERIES)
    template = random.choice(SEARCH_TEMPLATES)
    prompt = template.format(query=query)
    completion = json.dumps({"tool": "search_web", "args": {"query": query}})
    return {"prompt": prompt, "completion": completion}


def generate_dataset(num_examples: int = 500):
    examples = []
    per_tool = num_examples // 3
    generators = [generate_email_example, generate_weather_example, generate_search_example]

    for gen in generators:
        for _ in range(per_tool):
            examples.append(gen())

    while len(examples) < num_examples:
        gen = random.choice(generators)
        examples.append(gen())

    random.shuffle(examples)
    return examples


if __name__ == "__main__":
    print("Generating synthetic training dataset...")

    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    examples = generate_dataset(500)

    output_path = data_dir / "tool_calls.jsonl"
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Created {len(examples)} training examples")
    print(f"Saved to: {output_path}")

    print("\nSample examples:")
    print("-" * 60)
    for ex in examples[:5]:
        print(f"PROMPT:     {ex['prompt']}")
        print(f"COMPLETION: {ex['completion']}")
        print("-" * 60)

    tool_counts = {"send_email": 0, "search_web": 0, "get_weather": 0}
    for ex in examples:
        tool = json.loads(ex["completion"])["tool"]
        tool_counts[tool] += 1

    print("\nTool distribution:")
    for tool, count in tool_counts.items():
        print(f"  {tool}: {count}")
