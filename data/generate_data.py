"""
Synthetic training data generator for Pocket-Agent.
Uses GPT-4o-mini as teacher LLM to generate 1,500 diverse examples.

Usage:
    export OPENAI_API_KEY=sk-...
    python data/generate_data.py [--count 1500] [--out data/training_data.jsonl]
"""

import argparse
import json
import os
import re
import hashlib
import random
import time
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("pip install openai")

ROOT = Path(__file__).parent.parent
SCHEMAS = json.loads((ROOT / "data" / "tool_schemas.json").read_text())

SYSTEM_PROMPT = """\
You are Pocket-Agent, a compact mobile assistant. You have access to exactly these five tools:

weather   – args: location (string), unit ("C" or "F")
calendar  – args: action ("list" or "create"), date (YYYY-MM-DD), title (string, only for create)
convert   – args: value (number), from_unit (string), to_unit (string)
currency  – args: amount (number), from (ISO-4217 code), to (ISO-4217 code)
sql       – args: query (string)

Rules (follow exactly):
1. If the user's request clearly maps to one of the five tools, respond with ONLY a <tool_call> block:
   <tool_call>
   {"tool": "<name>", "args": {<args>}}
   </tool_call>
2. If the conversation has prior turns, resolve references ("that", "it", "there") against history.
3. If no tool fits — chitchat, ambiguous reference with no history, impossible request, or a tool that doesn't exist — respond in plain natural language WITHOUT any <tool_call> block.
4. Never invent tool names. Never emit partial JSON. Never add commentary outside the <tool_call> block when a tool call is appropriate.\
"""

# ---------------------------------------------------------------------------
# Prompt templates for each data slice
# ---------------------------------------------------------------------------

SLICE_PROMPTS = {
    "weather_single": [
        "What's the weather like in {city}?",
        "Tell me the temperature in {city} in {unit}.",
        "Is it hot in {city} right now? Give Celsius.",
        "Weather forecast for {city}?",
        "How's the weather in {city}, Fahrenheit please.",
        "Current conditions in {city}?",
        "What's the temp in {city} in °{unit}?",
        "Check weather for {city}.",
        "Give me weather for {city} in {unit}.",
        "اخبرني عن الطقس في {city}",  # Arabic
        "{city} mein aaj mausam kaisa hai?",  # Urdu/Hindi
        "¿Cómo está el tiempo en {city}?",  # Spanish
    ],
    "calendar_list": [
        "What's on my calendar for {date}?",
        "Show me my schedule on {date}.",
        "List events for {date}.",
        "Do I have anything planned for {date}?",
        "What meetings do I have on {date}?",
        "Check my agenda for {date}.",
    ],
    "calendar_create": [
        "Schedule '{title}' for {date}.",
        "Add '{title}' to my calendar on {date}.",
        "Create a meeting called '{title}' on {date}.",
        "Book '{title}' for {date}.",
        "Put '{title}' in my calendar on {date}.",
        "Set up '{title}' event for {date}.",
    ],
    "convert_single": [
        "Convert {value} {from_unit} to {to_unit}.",
        "How many {to_unit} is {value} {from_unit}?",
        "What is {value} {from_unit} in {to_unit}?",
        "{value} {from_unit} to {to_unit} please.",
        "I need {value} {from_unit} converted to {to_unit}.",
        "Turn {value} {from_unit} into {to_unit}.",
    ],
    "currency_single": [
        "Convert {amount} {from_curr} to {to_curr}.",
        "How much is {amount} {from_curr} in {to_curr}?",
        "Exchange {amount} {from_curr} to {to_curr}.",
        "What's {amount} {from_curr} in {to_curr}?",
        "{amount} {from_curr} equals how many {to_curr}?",
        "I have {amount} {from_curr}, convert to {to_curr}.",
    ],
    "sql_single": [
        "Run this query: {query}",
        "Execute: {query}",
        "Query the database: {query}",
        "Run SQL: {query}",
        "{query}",
    ],
    "refusal_chitchat": [
        "What's your name?",
        "Tell me a joke.",
        "How are you?",
        "What can you do?",
        "Thanks for your help!",
        "You're great!",
        "What's 2 + 2?",
        "Who won the World Cup in 2022?",
        "Explain quantum computing.",
        "Write me a poem.",
        "What's the meaning of life?",
        "Recommend a good book.",
    ],
    "refusal_unknown_tool": [
        "Send an email to my boss.",
        "Play some music.",
        "Take a photo.",
        "Turn on the lights.",
        "Order me a pizza.",
        "Call my mom.",
        "Search the web for cats.",
        "Post this to Twitter: Hello world",
        "Set a timer for 10 minutes.",
        "Open Spotify.",
    ],
    "refusal_ambiguous": [
        "Convert that.",
        "What about the other one?",
        "And tomorrow?",
        "Change it.",
        "Show me more.",
        "What's there?",
    ],
}

# Sample data pools
CITIES = [
    "London", "Tokyo", "New York", "Paris", "Dubai", "Sydney", "Berlin",
    "Mumbai", "Toronto", "Singapore", "Karachi", "Cairo", "Lagos", "Seoul",
    "Mexico City", "São Paulo", "Istanbul", "Moscow", "Bangkok", "Jakarta",
]
DATES = [
    "2025-01-15", "2025-02-20", "2025-03-10", "2025-04-05", "2025-05-30",
    "2025-06-15", "2025-07-04", "2025-08-12", "2025-09-25", "2025-10-31",
    "2025-11-11", "2025-12-25", "2026-01-01", "2026-02-14", "2026-03-08",
]
TITLES = [
    "Team standup", "Doctor appointment", "Lunch with Alex", "Board meeting",
    "Project review", "Dentist visit", "Birthday party", "Sales call",
    "Training session", "Quarterly review", "Client meeting", "Job interview",
]
CONVERT_PAIRS = [
    (5, "km", "miles"), (100, "lbs", "kg"), (72, "F", "C"), (30, "C", "F"),
    (1.5, "liters", "gallons"), (200, "cm", "inches"), (10, "oz", "grams"),
    (50, "mph", "kph"), (1000, "meters", "feet"), (2.5, "acres", "hectares"),
    (500, "ml", "cups"), (3, "yards", "meters"), (15, "stone", "kg"),
    (100, "calories", "kilojoules"), (60, "minutes", "hours"),
]
CURRENCY_PAIRS = [
    (100, "USD", "EUR"), (50, "EUR", "GBP"), (1000, "JPY", "USD"),
    (500, "GBP", "INR"), (200, "AUD", "CAD"), (75, "CHF", "USD"),
    (1500, "INR", "USD"), (300, "CAD", "EUR"), (10000, "PKR", "USD"),
    (250, "SAR", "EUR"), (1000, "AED", "USD"), (800, "SGD", "USD"),
]
SQL_QUERIES = [
    "SELECT * FROM users WHERE active = 1",
    "SELECT COUNT(*) FROM orders WHERE date > '2025-01-01'",
    "SELECT name, email FROM customers ORDER BY name",
    "SELECT SUM(amount) FROM transactions WHERE type = 'credit'",
    "SELECT * FROM products WHERE category = 'electronics' LIMIT 10",
    "UPDATE users SET last_login = NOW() WHERE id = 42",
    "SELECT AVG(salary) FROM employees WHERE department = 'Engineering'",
    "SELECT * FROM logs WHERE level = 'ERROR' ORDER BY timestamp DESC LIMIT 50",
    "SELECT product_id, COUNT(*) as views FROM page_views GROUP BY product_id",
    "SELECT u.name, COUNT(o.id) as orders FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id",
]

ADVERSARIAL_PROMPTS = [
    # Typos
    "wether in {city} celsius plz",
    "convrt {value} {from_unit} to {to_unit}",
    "curency {amount} {from_curr} to {to_curr}",
    "calander for {date}",
    "schedul '{title}' on {date}",
    # Code-switched (Hindi/Urdu + English)
    "{city} ka weather Celsius mein batao",
    "Mujhe {amount} {from_curr} ko {to_curr} mein convert karo",
    "{value} {from_unit} ko {to_unit} mein badlo",
    # Code-switched (Spanish + English)
    "¿Cuál es el tiempo en {city} en Fahrenheit?",
    "Convierte {value} {from_unit} a {to_unit} por favor",
    # Unit ambiguity
    "Convert {value} C to F",  # C could be cups or Celsius
    "Convert {value} oz to ml",  # fluid vs weight oz
    "Weather in {city}",  # no unit specified - model should pick default or ask
    # Hallucination-bait (non-existent places/currencies)
    "Weather in Narnia",
    "Convert 100 USD to Wakandan Vibranium",
    "Weather in {city} in Kelvin",  # K not in schema
]

MULTITURN_TEMPLATES = [
    {
        "turns": [
            {"role": "user", "content": "What's {amount} {from_curr} in {to_curr}?"},
            {"role": "assistant", "content": "<tool_call>\n{{\"tool\": \"currency\", \"args\": {{\"amount\": {amount}, \"from\": \"{from_curr}\", \"to\": \"{to_curr}\"}}}}\n</tool_call>"},
            {"role": "user", "content": "Now convert that to {third_curr}."},
            {"role": "assistant", "content": "<tool_call>\n{{\"tool\": \"currency\", \"args\": {{\"amount\": {amount}, \"from\": \"{from_curr}\", \"to\": \"{third_curr}\"}}}}\n</tool_call>"},
        ]
    },
    {
        "turns": [
            {"role": "user", "content": "Weather in {city}?"},
            {"role": "assistant", "content": "<tool_call>\n{{\"tool\": \"weather\", \"args\": {{\"location\": \"{city}\", \"unit\": \"C\"}}}}\n</tool_call>"},
            {"role": "user", "content": "And what about tomorrow's calendar?"},
            {"role": "assistant", "content": "<tool_call>\n{{\"tool\": \"calendar\", \"args\": {{\"action\": \"list\", \"date\": \"{date}\"}}}}\n</tool_call>"},
        ]
    },
    {
        "turns": [
            {"role": "user", "content": "Convert {value} {from_unit} to {to_unit}."},
            {"role": "assistant", "content": "<tool_call>\n{{\"tool\": \"convert\", \"args\": {{\"value\": {value}, \"from_unit\": \"{from_unit}\", \"to_unit\": \"{to_unit}\"}}}}\n</tool_call>"},
            {"role": "user", "content": "Now convert it to {to_unit2} instead."},
            {"role": "assistant", "content": "<tool_call>\n{{\"tool\": \"convert\", \"args\": {{\"value\": {value}, \"from_unit\": \"{from_unit}\", \"to_unit\": \"{to_unit2}\"}}}}\n</tool_call>"},
        ]
    },
    {
        "turns": [
            {"role": "user", "content": "Schedule '{title}' for {date}."},
            {"role": "assistant", "content": "<tool_call>\n{{\"tool\": \"calendar\", \"args\": {{\"action\": \"create\", \"date\": \"{date}\", \"title\": \"{title}\"}}}}\n</tool_call>"},
            {"role": "user", "content": "Actually what do I have that day?"},
            {"role": "assistant", "content": "<tool_call>\n{{\"tool\": \"calendar\", \"args\": {{\"action\": \"list\", \"date\": \"{date}\"}}}}\n</tool_call>"},
        ]
    },
]


def fill_template(template: str) -> str:
    city = random.choice(CITIES)
    date = random.choice(DATES)
    title = random.choice(TITLES)
    cv, fu, tu = random.choice(CONVERT_PAIRS)
    am, fc, tc = random.choice(CURRENCY_PAIRS)
    query = random.choice(SQL_QUERIES)
    unit = random.choice(["C", "F"])

    return template.format(
        city=city, date=date, title=title,
        value=cv, from_unit=fu, to_unit=tu,
        amount=am, from_curr=fc, to_curr=tc,
        query=query, unit=unit,
    )


def make_single_turn_example(user_msg: str, expected_tool: str | None, expected_args: dict | None) -> dict:
    if expected_tool:
        assistant_content = f"<tool_call>\n{json.dumps({'tool': expected_tool, 'args': expected_args})}\n</tool_call>"
    else:
        # For refusals we ask GPT to generate a natural response — handled separately
        assistant_content = None

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_msg})
    if assistant_content:
        messages.append({"role": "assistant", "content": assistant_content})
    return {"messages": messages}


def generate_rule_based_examples() -> list[dict]:
    """Fast, free, deterministic examples from templates."""
    examples = []

    # Weather
    for _ in range(220):
        city = random.choice(CITIES)
        unit = random.choice(["C", "F"])
        template = random.choice(SLICE_PROMPTS["weather_single"])
        try:
            user_msg = fill_template(template).replace("{city}", city).replace("{unit}", unit)
        except Exception:
            user_msg = f"What's the weather in {city} in {unit}?"
        ex = make_single_turn_example(user_msg, "weather", {"location": city, "unit": unit})
        examples.append(ex)

    # Calendar list
    for _ in range(100):
        date = random.choice(DATES)
        template = random.choice(SLICE_PROMPTS["calendar_list"])
        user_msg = template.format(date=date)
        ex = make_single_turn_example(user_msg, "calendar", {"action": "list", "date": date})
        examples.append(ex)

    # Calendar create
    for _ in range(100):
        date = random.choice(DATES)
        title = random.choice(TITLES)
        template = random.choice(SLICE_PROMPTS["calendar_create"])
        user_msg = template.format(title=title, date=date)
        ex = make_single_turn_example(user_msg, "calendar", {"action": "create", "date": date, "title": title})
        examples.append(ex)

    # Convert
    for _ in range(220):
        cv, fu, tu = random.choice(CONVERT_PAIRS)
        template = random.choice(SLICE_PROMPTS["convert_single"])
        user_msg = template.format(value=cv, from_unit=fu, to_unit=tu)
        ex = make_single_turn_example(user_msg, "convert", {"value": cv, "from_unit": fu, "to_unit": tu})
        examples.append(ex)

    # Currency
    for _ in range(220):
        am, fc, tc = random.choice(CURRENCY_PAIRS)
        template = random.choice(SLICE_PROMPTS["currency_single"])
        user_msg = template.format(amount=am, from_curr=fc, to_curr=tc)
        ex = make_single_turn_example(user_msg, "currency", {"amount": am, "from": fc, "to": tc})
        examples.append(ex)

    # SQL
    for _ in range(100):
        query = random.choice(SQL_QUERIES)
        template = random.choice(SLICE_PROMPTS["sql_single"])
        user_msg = template.format(query=query)
        ex = make_single_turn_example(user_msg, "sql", {"query": query})
        examples.append(ex)

    # Multi-turn (rule-based)
    for _ in range(150):
        tmpl = random.choice(MULTITURN_TEMPLATES)
        city = random.choice(CITIES)
        date = random.choice(DATES)
        title = random.choice(TITLES)
        cv, fu, tu = random.choice(CONVERT_PAIRS)
        _, tu2, _ = random.choice(CONVERT_PAIRS)
        am, fc, tc = random.choice(CURRENCY_PAIRS)
        _, _, third_curr = random.choice(CURRENCY_PAIRS)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for turn in tmpl["turns"]:
            content = turn["content"].format(
                city=city, date=date, title=title,
                value=cv, from_unit=fu, to_unit=tu, to_unit2=tu2,
                amount=am, from_curr=fc, to_curr=tc, third_curr=third_curr,
            )
            messages.append({"role": turn["role"], "content": content})
        examples.append({"messages": messages})

    return examples


def generate_refusal_examples_via_api(client: OpenAI, count: int = 150) -> list[dict]:
    """Ask GPT-4o-mini to produce natural refusal responses."""
    examples = []
    prompts_pool = (
        SLICE_PROMPTS["refusal_chitchat"]
        + SLICE_PROMPTS["refusal_unknown_tool"]
        + SLICE_PROMPTS["refusal_ambiguous"]
    )

    batch_size = 10
    batches_needed = (count + batch_size - 1) // batch_size

    for batch_idx in range(batches_needed):
        batch_prompts = random.choices(prompts_pool, k=batch_size)
        prompt_list = "\n".join(f"{i+1}. {p}" for i, p in enumerate(batch_prompts))

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are generating training data for a mobile assistant that can ONLY use these tools: "
                            "weather, calendar, convert, currency, sql.\n\n"
                            "For each user message below, write a short natural assistant response that does NOT "
                            "use any tool call — because the request doesn't match any tool, is ambiguous, or is chitchat.\n\n"
                            "Return a JSON array of strings, one response per item. Match the order of inputs.\n"
                            "Keep responses concise (1-2 sentences)."
                        ),
                    },
                    {"role": "user", "content": f"User messages:\n{prompt_list}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.8,
            )
            data = json.loads(response.choices[0].message.content)
            responses = data.get("responses", list(data.values())[0] if data else [])

            for user_msg, asst_resp in zip(batch_prompts, responses):
                if isinstance(asst_resp, str) and asst_resp.strip():
                    examples.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": asst_resp.strip()},
                        ]
                    })
        except Exception as e:
            print(f"  [refusal batch {batch_idx}] error: {e}")
            time.sleep(2)

        if len(examples) >= count:
            break

    return examples[:count]


def generate_adversarial_via_api(client: OpenAI, count: int = 150) -> list[dict]:
    """GPT-4o-mini generates correct responses for typo/code-switched inputs."""
    examples = []

    for _ in range(count):
        city = random.choice(CITIES)
        date = random.choice(DATES)
        cv, fu, tu = random.choice(CONVERT_PAIRS)
        am, fc, tc = random.choice(CURRENCY_PAIRS)
        title = random.choice(TITLES)

        template = random.choice(ADVERSARIAL_PROMPTS)
        try:
            user_msg = template.format(
                city=city, date=date, title=title,
                value=cv, from_unit=fu, to_unit=tu,
                amount=am, from_curr=fc, to_curr=tc,
            )
        except KeyError:
            user_msg = template  # pre-filled template

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.2,
                max_tokens=150,
            )
            asst_content = response.choices[0].message.content.strip()

            # Validate: if tool call, must be parseable
            if "<tool_call>" in asst_content:
                match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", asst_content, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group(1))
                        if parsed.get("tool") not in {"weather", "calendar", "convert", "currency", "sql"}:
                            continue
                    except json.JSONDecodeError:
                        continue

            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": asst_content},
                ]
            })
        except Exception as e:
            print(f"  [adversarial] error: {e}")
            time.sleep(1)

    return examples


def validate_dataset(examples: list[dict]) -> tuple[list[dict], int]:
    """Filter out malformed tool calls."""
    valid = []
    rejected = 0
    valid_tools = {"weather", "calendar", "convert", "currency", "sql"}

    for ex in examples:
        messages = ex.get("messages", [])
        ok = True
        for msg in messages:
            if msg["role"] == "assistant":
                content = msg["content"]
                if "<tool_call>" in content:
                    match = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", content, re.DOTALL)
                    if not match:
                        ok = False
                        break
                    try:
                        parsed = json.loads(match.group(1))
                        if parsed.get("tool") not in valid_tools:
                            ok = False
                            break
                    except json.JSONDecodeError:
                        ok = False
                        break
        if ok:
            valid.append(ex)
        else:
            rejected += 1

    return valid, rejected


def check_no_overlap_with_public_test(examples: list[dict], public_test_path: Path) -> None:
    """SHA-256 check: ensure no training prompt matches public test set."""
    if not public_test_path.exists():
        print("  [overlap check] public_test.jsonl not found — skipping")
        return

    public_hashes = set()
    for line in public_test_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        user_msg = ""
        for m in item.get("messages", []):
            if m["role"] == "user":
                user_msg = m["content"]
                break
        public_hashes.add(hashlib.sha256(user_msg.encode()).hexdigest())

    overlaps = 0
    for ex in examples:
        for m in ex.get("messages", []):
            if m["role"] == "user":
                h = hashlib.sha256(m["content"].encode()).hexdigest()
                if h in public_hashes:
                    overlaps += 1
                break

    if overlaps:
        print(f"  WARNING: {overlaps} training examples overlap with public test set!")
    else:
        print("  [overlap check] PASSED — zero overlap with public test set")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1500)
    parser.add_argument("--out", default="data/training_data.jsonl")
    parser.add_argument("--no-api", action="store_true", help="Skip API calls (rule-based only)")
    args = parser.parse_args()

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.no_api:
        raise SystemExit("Set OPENAI_API_KEY or use --no-api for rule-based generation only")

    client = OpenAI(api_key=api_key) if api_key else None

    print("=== Pocket-Agent Data Generation ===")
    print(f"Target: {args.count} examples → {out_path}")

    all_examples: list[dict] = []

    # Phase 1: Rule-based (free, fast)
    print("\n[1/3] Generating rule-based examples...")
    rule_based = generate_rule_based_examples()
    print(f"  Generated {len(rule_based)} rule-based examples")
    all_examples.extend(rule_based)

    if not args.no_api and client:
        # Phase 2: Refusals via API
        print("\n[2/3] Generating refusal examples via GPT-4o-mini...")
        refusals = generate_refusal_examples_via_api(client, count=200)
        print(f"  Generated {len(refusals)} refusal examples")
        all_examples.extend(refusals)

        # Phase 3: Adversarial via API
        print("\n[3/3] Generating adversarial examples via GPT-4o-mini...")
        adversarial = generate_adversarial_via_api(client, count=150)
        print(f"  Generated {len(adversarial)} adversarial examples")
        all_examples.extend(adversarial)
    else:
        print("\n[2/3] Skipping API calls (--no-api flag set)")
        print("[3/3] Adding rule-based refusal stubs...")
        # Rule-based refusals without response text (model will learn from context)
        for _ in range(200):
            user_msg = random.choice(
                SLICE_PROMPTS["refusal_chitchat"] + SLICE_PROMPTS["refusal_unknown_tool"]
            )
            all_examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": "I'm sorry, I can only help with weather, calendar, unit conversion, currency exchange, and SQL queries."},
                ]
            })

    # Validate
    print("\n[Validation] Checking tool call JSON integrity...")
    valid_examples, rejected = validate_dataset(all_examples)
    print(f"  Valid: {len(valid_examples)}, Rejected: {rejected}")

    # Shuffle
    random.shuffle(valid_examples)

    # Overlap check
    print("[Validation] Checking overlap with public test set...")
    check_no_overlap_with_public_test(valid_examples, ROOT / "data" / "public_test.jsonl")

    # Write output
    with out_path.open("w") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n✓ Wrote {len(valid_examples)} examples to {out_path}")
    print("\nDistribution:")
    tool_counts: dict[str, int] = {}
    refusal_count = 0
    for ex in valid_examples:
        found_tool = False
        for msg in ex["messages"]:
            if msg["role"] == "assistant" and "<tool_call>" in msg["content"]:
                m = re.search(r'"tool"\s*:\s*"([^"]+)"', msg["content"])
                if m:
                    t = m.group(1)
                    tool_counts[t] = tool_counts.get(t, 0) + 1
                    found_tool = True
                    break
        if not found_tool:
            refusal_count += 1
    for tool, cnt in sorted(tool_counts.items()):
        print(f"  {tool}: {cnt}")
    print(f"  refusals: {refusal_count}")


if __name__ == "__main__":
    main()
