"""
Synthetic training data generator for Pocket-Agent.
Uses GPT-4o-mini as teacher LLM to generate ~1,500 diverse, correct examples.

Usage:
    export OPENAI_API_KEY=sk-...
    python data/generate_data.py [--count 1500] [--out data/training_data.jsonl]
    python data/generate_data.py --no-api   # free, rule-based only
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

# ── System prompt ─────────────────────────────────────────────────────────────
# This exact prompt is used BOTH in training data AND at inference time.
# Precision here directly determines grading accuracy.

SYSTEM_PROMPT = """\
You are Pocket-Agent, a compact offline mobile assistant. You have access to exactly five tools:

  weather   – Get current weather.
              Args: location (city/place as a string), unit ("C" for Celsius or "F" for Fahrenheit)
  calendar  – Manage calendar events.
              Args: action ("list" to show events, "create" to add one), date (YYYY-MM-DD), title (string — required only when action is "create")
  convert   – Convert between units (length, weight, temperature, volume, speed, area, etc.).
              Args: value (number), from_unit (string), to_unit (string)
  currency  – Convert between currencies using live rates.
              Args: amount (number), from (ISO-4217 3-letter code e.g. "USD"), to (ISO-4217 3-letter code e.g. "EUR")
  sql       – Execute a SQL query against the local database.
              Args: query (valid SQL string)

Output format when calling a tool — use EXACTLY this, no other text:
<tool_call>
{"tool": "<tool_name>", "args": {<args_as_json>}}
</tool_call>

Rules — follow with zero deviation:
1. TOOL CALL: When the user's request clearly maps to one of the five tools, output ONLY the <tool_call> block. No preamble, no explanation, no text before or after.
2. DEFAULT UNIT: For weather requests that do not specify a unit, always default to "C" (Celsius).
3. MULTI-TURN RESOLUTION: When the user uses references like "that", "it", "there", "same amount", "convert that", resolve the referent from conversation history and use the resolved values in args.
4. REFUSAL — respond in plain natural text (NO <tool_call> block) when:
   - The request is chitchat, a general knowledge question, or small talk
   - The request requires a tool that does not exist (email, timer, music, maps, camera, etc.)
   - The reference ("that", "it") cannot be resolved because there is no prior context
   - The request is fundamentally impossible or nonsensical
5. NEVER invent tool names outside the five listed. NEVER output partial or malformed JSON. NEVER wrap a refusal in a <tool_call> block.\
"""

# ── Data pools ────────────────────────────────────────────────────────────────

CITIES = [
    "London", "Tokyo", "New York", "Paris", "Dubai", "Sydney", "Berlin",
    "Mumbai", "Toronto", "Singapore", "Karachi", "Cairo", "Lagos", "Seoul",
    "Mexico City", "São Paulo", "Istanbul", "Moscow", "Bangkok", "Jakarta",
    "Lahore", "Dhaka", "Tehran", "Baghdad", "Riyadh", "Nairobi", "Casablanca",
    "Buenos Aires", "Lima", "Bogotá", "Amsterdam", "Rome", "Madrid", "Vienna",
]

DATES = [
    "2025-01-15", "2025-02-20", "2025-03-10", "2025-04-05", "2025-05-30",
    "2025-06-15", "2025-07-04", "2025-08-12", "2025-09-25", "2025-10-31",
    "2025-11-11", "2025-12-25", "2026-01-01", "2026-02-14", "2026-03-08",
    "2026-04-18", "2026-05-01", "2026-06-20", "2026-07-10", "2026-08-22",
]

TITLES = [
    "Team standup", "Doctor appointment", "Lunch with Alex", "Board meeting",
    "Project review", "Dentist visit", "Birthday party", "Sales call",
    "Training session", "Quarterly review", "Client meeting", "Job interview",
    "Flight to Dubai", "Gym session", "Product launch", "1:1 with manager",
    "School pickup", "Vet appointment", "Conference call", "Sprint planning",
]

CONVERT_PAIRS = [
    # (value, from_unit, to_unit)
    (5, "km", "miles"),
    (100, "lbs", "kg"),
    (72, "F", "C"),
    (30, "C", "F"),
    (1.5, "liters", "gallons"),
    (200, "cm", "inches"),
    (10, "oz", "grams"),
    (50, "mph", "kph"),
    (1000, "meters", "feet"),
    (2.5, "acres", "hectares"),
    (500, "ml", "cups"),
    (3, "yards", "meters"),
    (15, "stone", "kg"),
    (100, "calories", "kilojoules"),
    (60, "minutes", "hours"),
    (8, "feet", "meters"),
    (250, "grams", "oz"),
    (10, "miles", "km"),
    (37, "C", "F"),
    (98.6, "F", "C"),
    (5, "kg", "lbs"),
    (1, "mile", "km"),
    (100, "kph", "mph"),
    (1000, "grams", "kg"),
]

CURRENCY_PAIRS = [
    # (amount, from_ISO, to_ISO)
    (100, "USD", "EUR"),
    (50, "EUR", "GBP"),
    (1000, "JPY", "USD"),
    (500, "GBP", "INR"),
    (200, "AUD", "CAD"),
    (75, "CHF", "USD"),
    (1500, "INR", "USD"),
    (300, "CAD", "EUR"),
    (10000, "PKR", "USD"),
    (250, "SAR", "EUR"),
    (1000, "AED", "USD"),
    (800, "SGD", "USD"),
    (5000, "MXN", "USD"),
    (200, "USD", "JPY"),
    (1000, "USD", "PKR"),
    (500, "EUR", "GBP"),
    (100, "GBP", "USD"),
    (2000, "INR", "PKR"),
    (150, "USD", "AED"),
    (300, "USD", "SAR"),
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
    "DELETE FROM sessions WHERE expires_at < NOW()",
    "SELECT * FROM inventory WHERE quantity < 10",
    "SELECT DISTINCT city FROM addresses",
    "INSERT INTO events (title, date) VALUES ('Meeting', '2025-06-15')",
    "SELECT * FROM employees WHERE hire_date > '2024-01-01' ORDER BY hire_date",
]

CURRENCY_NAMES = {
    "USD": ["dollars", "US dollars", "American dollars", "USD"],
    "EUR": ["euros", "Euro", "EUR"],
    "GBP": ["pounds", "British pounds", "sterling", "GBP"],
    "INR": ["rupees", "Indian rupees", "INR"],
    "PKR": ["rupees", "Pakistani rupees", "PKR"],
    "JPY": ["yen", "Japanese yen", "JPY"],
    "AED": ["dirhams", "UAE dirhams", "AED"],
    "SAR": ["riyals", "Saudi riyals", "SAR"],
    "CAD": ["Canadian dollars", "CAD"],
    "AUD": ["Australian dollars", "AUD"],
    "CHF": ["Swiss francs", "francs", "CHF"],
    "SGD": ["Singapore dollars", "SGD"],
    "MXN": ["pesos", "Mexican pesos", "MXN"],
}


def currency_name(iso: str) -> str:
    names = CURRENCY_NAMES.get(iso, [iso])
    return random.choice(names)


# ── Per-tool template generators ──────────────────────────────────────────────
# Each function produces (user_msg, ground_truth_args) pairs with guaranteed consistency.
# NO fill_template() calls — every variable is set once and used consistently.

def gen_weather(n: int = 220) -> list[dict]:
    """Generate weather examples with correct city/unit pairing."""
    examples = []
    for _ in range(n):
        city = random.choice(CITIES)
        r = random.random()

        if r < 0.22:
            # Explicit Celsius in prompt → unit = "C"
            unit = "C"
            user_msg = random.choice([
                f"What's the weather in {city} in Celsius?",
                f"How hot is it in {city}? Celsius please.",
                f"Temperature in {city} in degrees Celsius?",
                f"{city} weather, Celsius.",
                f"Give me the temperature in {city} in °C.",
                f"Current temp in {city} — Celsius.",
                f"Weather for {city} in C?",
            ])
        elif r < 0.42:
            # Explicit Fahrenheit in prompt → unit = "F"
            unit = "F"
            user_msg = random.choice([
                f"What's the weather in {city} in Fahrenheit?",
                f"How hot is it in {city}? Fahrenheit please.",
                f"Temperature in {city} in °F?",
                f"{city} weather, Fahrenheit.",
                f"What's the temp in {city} in Fahrenheit?",
                f"Weather for {city} in F?",
            ])
        elif r < 0.68:
            # No unit specified → always default to "C"
            unit = "C"
            user_msg = random.choice([
                f"What's the weather like in {city}?",
                f"Weather in {city}?",
                f"Weather forecast for {city}?",
                f"How's the weather in {city}?",
                f"Current conditions in {city}?",
                f"What's the temperature in {city}?",
                f"Is it hot in {city} right now?",
                f"Check the weather in {city}.",
                f"What should I wear in {city} today?",
                f"Tell me the weather in {city}.",
                # Multilingual — no unit → default "C"
                f"اخبرني عن الطقس في {city}",
                f"{city} mein aaj mausam kaisa hai?",
                f"¿Cómo está el tiempo en {city}?",
                f"{city} ka mausam batao",
                f"Bataiye {city} mein kaisa mausam hai",
                f"هوای {city} چطوره؟",
            ])
        else:
            # Unit as variable — choose unit first, then build prompt around it
            unit = random.choice(["C", "F"])
            unit_word = "Celsius" if unit == "C" else "Fahrenheit"
            user_msg = random.choice([
                f"What's the temp in {city} in {unit_word}?",
                f"Tell me the temperature in {city} in {unit}.",
                f"Give me weather for {city} in {unit}.",
                f"{city} weather in {unit_word} please.",
                f"Weather in {city} ({unit})?",
                f"I need the weather in {city} — {unit_word}.",
            ])

        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f'<tool_call>\n{{"tool": "weather", "args": {{"location": "{city}", "unit": "{unit}"}}}}\n</tool_call>'},
            ]
        })
    return examples


def gen_calendar_list(n: int = 110) -> list[dict]:
    examples = []
    for _ in range(n):
        date = random.choice(DATES)
        user_msg = random.choice([
            f"What's on my calendar for {date}?",
            f"Show me my schedule on {date}.",
            f"List events for {date}.",
            f"Do I have anything planned for {date}?",
            f"What meetings do I have on {date}?",
            f"Check my agenda for {date}.",
            f"Any appointments on {date}?",
            f"What's happening on {date}?",
            f"Show events on {date}.",
            f"Calendar for {date}?",
            f"{date} mein kya schedule hai?",
            f"¿Qué tengo en el calendario el {date}?",
            f"Kya hai mere calendar mein {date} ko?",
        ])
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f'<tool_call>\n{{"tool": "calendar", "args": {{"action": "list", "date": "{date}"}}}}\n</tool_call>'},
            ]
        })
    return examples


def gen_calendar_create(n: int = 110) -> list[dict]:
    examples = []
    for _ in range(n):
        date = random.choice(DATES)
        title = random.choice(TITLES)
        user_msg = random.choice([
            f"Schedule '{title}' for {date}.",
            f"Add '{title}' to my calendar on {date}.",
            f"Create a calendar event called '{title}' on {date}.",
            f"Book '{title}' for {date}.",
            f"Put '{title}' in my calendar on {date}.",
            f"Set up a '{title}' event for {date}.",
            f"Can you add {title} to my calendar on {date}?",
            f"New event: {title}, on {date}.",
            f"I have {title} on {date}, add it to my calendar.",
            f"Create: {title} — {date}.",
            f"{date} ko '{title}' calendar mein add karo.",
        ])
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f'<tool_call>\n{{"tool": "calendar", "args": {{"action": "create", "date": "{date}", "title": "{title}"}}}}\n</tool_call>'},
            ]
        })
    return examples


def gen_convert(n: int = 220) -> list[dict]:
    examples = []
    for _ in range(n):
        value, from_unit, to_unit = random.choice(CONVERT_PAIRS)
        user_msg = random.choice([
            f"Convert {value} {from_unit} to {to_unit}.",
            f"How many {to_unit} is {value} {from_unit}?",
            f"What is {value} {from_unit} in {to_unit}?",
            f"{value} {from_unit} to {to_unit} please.",
            f"I need {value} {from_unit} converted to {to_unit}.",
            f"Turn {value} {from_unit} into {to_unit}.",
            f"Convert: {value} {from_unit} → {to_unit}",
            f"What's {value} {from_unit} in {to_unit}?",
            f"How much is {value} {from_unit} in {to_unit}?",
            f"{value} {from_unit} equals how many {to_unit}?",
            f"I have {value} {from_unit}. What is that in {to_unit}?",
            # Code-switched
            f"{value} {from_unit} ko {to_unit} mein convert karo.",
            f"Convierte {value} {from_unit} a {to_unit}.",
        ])
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f'<tool_call>\n{{"tool": "convert", "args": {{"value": {value}, "from_unit": "{from_unit}", "to_unit": "{to_unit}"}}}}\n</tool_call>'},
            ]
        })
    return examples


def gen_currency(n: int = 220) -> list[dict]:
    examples = []
    for _ in range(n):
        amount, from_iso, to_iso = random.choice(CURRENCY_PAIRS)
        from_name = currency_name(from_iso)
        to_name = currency_name(to_iso)
        user_msg = random.choice([
            f"Convert {amount} {from_iso} to {to_iso}.",
            f"How much is {amount} {from_iso} in {to_iso}?",
            f"Exchange {amount} {from_iso} to {to_iso}.",
            f"What's {amount} {from_iso} in {to_iso}?",
            f"{amount} {from_iso} equals how many {to_iso}?",
            f"I have {amount} {from_iso}, convert to {to_iso}.",
            f"What is {amount} {from_name} in {to_name}?",
            f"Convert {amount} {from_name} to {to_name}.",
            f"How many {to_name} is {amount} {from_name}?",
            f"{amount} {from_name} to {to_iso}.",
            # Code-switched
            f"Mujhe {amount} {from_iso} ko {to_iso} mein convert karo.",
            f"{amount} {from_iso} ko {to_iso} mein badlo.",
            f"¿Cuánto son {amount} {from_iso} en {to_iso}?",
        ])
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f'<tool_call>\n{{"tool": "currency", "args": {{"amount": {amount}, "from": "{from_iso}", "to": "{to_iso}"}}}}\n</tool_call>'},
            ]
        })
    return examples


def gen_sql(n: int = 110) -> list[dict]:
    examples = []
    for _ in range(n):
        query = random.choice(SQL_QUERIES)
        user_msg = random.choice([
            f"Run this query: {query}",
            f"Execute: {query}",
            f"Query the database: {query}",
            f"Run SQL: {query}",
            f"Run this: {query}",
            f"Execute this SQL: {query}",
            f"Database query: {query}",
            f"{query}",
        ])
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f'<tool_call>\n{{"tool": "sql", "args": {{"query": "{query}"}}}}\n</tool_call>'},
            ]
        })
    return examples


# ── Multi-turn examples ───────────────────────────────────────────────────────

def gen_multiturn(n: int = 180) -> list[dict]:
    """
    Multi-turn conversations where the model must resolve references.
    All variable bindings are consistent within each example.
    """
    examples = []

    templates = [
        # Currency → convert that to a third currency
        lambda: _mt_currency_chain(),
        # Weather → then calendar
        lambda: _mt_weather_then_calendar(),
        # Convert → change the target unit
        lambda: _mt_convert_change_unit(),
        # Calendar create → then list that day
        lambda: _mt_calendar_create_then_list(),
        # Currency → ask about the same pair reversed
        lambda: _mt_currency_then_reverse(),
        # Weather → ask in the other unit
        lambda: _mt_weather_unit_switch(),
        # SQL → another query on the same table
        lambda: _mt_sql_follow_up(),
    ]

    for _ in range(n):
        fn = random.choice(templates)
        try:
            ex = fn()
            if ex:
                examples.append(ex)
        except Exception:
            pass

    return examples


def _mt_currency_chain():
    amount, from_iso, to_iso = random.choice(CURRENCY_PAIRS)
    _, _, third_iso = random.choice([p for p in CURRENCY_PAIRS if p[2] != to_iso])
    ref = random.choice(["that", "the same amount", "it", "those"])
    turns = [
        {"role": "user", "content": f"What's {amount} {from_iso} in {to_iso}?"},
        {"role": "assistant", "content": f'<tool_call>\n{{"tool": "currency", "args": {{"amount": {amount}, "from": "{from_iso}", "to": "{to_iso}"}}}}\n</tool_call>'},
        {"role": "user", "content": f"Now convert {ref} to {third_iso}."},
        {"role": "assistant", "content": f'<tool_call>\n{{"tool": "currency", "args": {{"amount": {amount}, "from": "{from_iso}", "to": "{third_iso}"}}}}\n</tool_call>'},
    ]
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + turns}


def _mt_weather_then_calendar():
    city = random.choice(CITIES)
    date = random.choice(DATES)
    turns = [
        {"role": "user", "content": f"Weather in {city}?"},
        {"role": "assistant", "content": f'<tool_call>\n{{"tool": "weather", "args": {{"location": "{city}", "unit": "C"}}}}\n</tool_call>'},
        {"role": "user", "content": f"And what do I have on my calendar for {date}?"},
        {"role": "assistant", "content": f'<tool_call>\n{{"tool": "calendar", "args": {{"action": "list", "date": "{date}"}}}}\n</tool_call>'},
    ]
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + turns}


def _mt_convert_change_unit():
    value, from_unit, to_unit = random.choice(CONVERT_PAIRS)
    _, _, to_unit2 = random.choice([p for p in CONVERT_PAIRS if p[0] == value and p[1] == from_unit and p[2] != to_unit] or [random.choice(CONVERT_PAIRS)])
    # Ensure to_unit2 is different from to_unit
    other_pairs = [p for p in CONVERT_PAIRS if p[1] == from_unit and p[2] != to_unit]
    if not other_pairs:
        return None
    _, _, to_unit2 = random.choice(other_pairs)
    ref = random.choice(["it", "the same value", "that"])
    turns = [
        {"role": "user", "content": f"Convert {value} {from_unit} to {to_unit}."},
        {"role": "assistant", "content": f'<tool_call>\n{{"tool": "convert", "args": {{"value": {value}, "from_unit": "{from_unit}", "to_unit": "{to_unit}"}}}}\n</tool_call>'},
        {"role": "user", "content": f"Now convert {ref} to {to_unit2} instead."},
        {"role": "assistant", "content": f'<tool_call>\n{{"tool": "convert", "args": {{"value": {value}, "from_unit": "{from_unit}", "to_unit": "{to_unit2}"}}}}\n</tool_call>'},
    ]
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + turns}


def _mt_calendar_create_then_list():
    date = random.choice(DATES)
    title = random.choice(TITLES)
    turns = [
        {"role": "user", "content": f"Add '{title}' to my calendar on {date}."},
        {"role": "assistant", "content": f'<tool_call>\n{{"tool": "calendar", "args": {{"action": "create", "date": "{date}", "title": "{title}"}}}}\n</tool_call>'},
        {"role": "user", "content": random.choice([
            "Actually, what else do I have that day?",
            "What's on my schedule for that day?",
            "Show me everything for that date.",
            "What's my full agenda for that day?",
        ])},
        {"role": "assistant", "content": f'<tool_call>\n{{"tool": "calendar", "args": {{"action": "list", "date": "{date}"}}}}\n</tool_call>'},
    ]
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + turns}


def _mt_currency_then_reverse():
    amount, from_iso, to_iso = random.choice(CURRENCY_PAIRS)
    reverse_amount = round(amount * 0.85, 2)  # approximate, doesn't matter
    turns = [
        {"role": "user", "content": f"How much is {amount} {from_iso} in {to_iso}?"},
        {"role": "assistant", "content": f'<tool_call>\n{{"tool": "currency", "args": {{"amount": {amount}, "from": "{from_iso}", "to": "{to_iso}"}}}}\n</tool_call>'},
        {"role": "user", "content": f"What about the other way — {amount} {to_iso} to {from_iso}?"},
        {"role": "assistant", "content": f'<tool_call>\n{{"tool": "currency", "args": {{"amount": {amount}, "from": "{to_iso}", "to": "{from_iso}"}}}}\n</tool_call>'},
    ]
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + turns}


def _mt_weather_unit_switch():
    city = random.choice(CITIES)
    turns = [
        {"role": "user", "content": f"What's the weather in {city} in Celsius?"},
        {"role": "assistant", "content": f'<tool_call>\n{{"tool": "weather", "args": {{"location": "{city}", "unit": "C"}}}}\n</tool_call>'},
        {"role": "user", "content": random.choice([
            "And in Fahrenheit?",
            "What's that in Fahrenheit?",
            "Now give it in F.",
            "Same but Fahrenheit please.",
        ])},
        {"role": "assistant", "content": f'<tool_call>\n{{"tool": "weather", "args": {{"location": "{city}", "unit": "F"}}}}\n</tool_call>'},
    ]
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + turns}


def _mt_sql_follow_up():
    query1 = random.choice(SQL_QUERIES)
    query2 = random.choice([q for q in SQL_QUERIES if q != query1])
    turns = [
        {"role": "user", "content": f"Run: {query1}"},
        {"role": "assistant", "content": f'<tool_call>\n{{"tool": "sql", "args": {{"query": "{query1}"}}}}\n</tool_call>'},
        {"role": "user", "content": f"Now run: {query2}"},
        {"role": "assistant", "content": f'<tool_call>\n{{"tool": "sql", "args": {{"query": "{query2}"}}}}\n</tool_call>'},
    ]
    return {"messages": [{"role": "system", "content": SYSTEM_PROMPT}] + turns}


# ── Adversarial templates (rule-based) ────────────────────────────────────────

def gen_adversarial_rulebased() -> list[dict]:
    """
    Typos, code-switched, and unit-ambiguous examples with correct ground truth.
    For hallucination-bait, the model must refuse.
    """
    examples = []

    # --- Typo + code-switched: still map to real tools ---
    for _ in range(60):
        city = random.choice(CITIES)
        unit = random.choice(["C", "F"])
        user_msg = random.choice([
            f"wether in {city} celsius plz",
            f"wheather {city}",
            f"Celsius temprature for {city}?",
            f"{city} ka temprature batao celsius mein",
            f"mujhe {city} ka mausam chahiye",
            f"¿Cómo está el tiempo en {city}? en grados Celsius",
            f"{city} mein aaj kitni garmi hai?",
            f"bata {city} weather",
        ])
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f'<tool_call>\n{{"tool": "weather", "args": {{"location": "{city}", "unit": "C"}}}}\n</tool_call>'},
            ]
        })

    for _ in range(40):
        value, from_unit, to_unit = random.choice(CONVERT_PAIRS)
        user_msg = random.choice([
            f"convrt {value} {from_unit} to {to_unit}",
            f"convertt {value} {from_unit} in {to_unit}",
            f"{value} {from_unit} se {to_unit} mein badlo",
            f"Convierte {value} {from_unit} a {to_unit}",
            f"konvert {value} {from_unit} to {to_unit}",
        ])
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f'<tool_call>\n{{"tool": "convert", "args": {{"value": {value}, "from_unit": "{from_unit}", "to_unit": "{to_unit}"}}}}\n</tool_call>'},
            ]
        })

    for _ in range(40):
        amount, from_iso, to_iso = random.choice(CURRENCY_PAIRS)
        from_name = currency_name(from_iso)
        to_name = currency_name(to_iso)
        user_msg = random.choice([
            f"curency {amount} {from_iso} to {to_iso}",
            f"{amount} {from_name} ko {to_name} mein convert karo",
            f"¿Cuánto son {amount} {from_name} en {to_name}?",
            f"Mujhe {amount} {from_iso} se {to_iso} mein chahiye",
            f"currancy {amount} {from_iso} to {to_iso}",
        ])
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f'<tool_call>\n{{"tool": "currency", "args": {{"amount": {amount}, "from": "{from_iso}", "to": "{to_iso}"}}}}\n</tool_call>'},
            ]
        })

    for _ in range(20):
        date = random.choice(DATES)
        user_msg = random.choice([
            f"calander for {date}",
            f"calender {date}",
            f"{date} ka schedule dikhao",
            f"¿Qué tengo el {date}?",
        ])
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f'<tool_call>\n{{"tool": "calendar", "args": {{"action": "list", "date": "{date}"}}}}\n</tool_call>'},
            ]
        })

    # --- Hallucination-bait: must REFUSE ---
    hallucination_refusals = [
        ("Weather in Narnia", "I'm sorry, Narnia isn't a real location — I can't retrieve weather data for it."),
        ("Weather in Middle Earth in Celsius", "Middle Earth is fictional, so I can't retrieve weather data for it."),
        ("Convert 100 USD to Wakandan Vibranium", "Wakandan Vibranium isn't a real currency, so I can't perform that conversion."),
        ("Weather in {city} in Kelvin".format(city=random.choice(CITIES)), "I can only provide weather in Celsius (C) or Fahrenheit (F), not Kelvin."),
        ("Exchange rate for Dogecoin to USD", "I don't have a currency tool that handles cryptocurrency — I only support standard ISO-4217 currencies."),
        ("Convert 10 furlongs per fortnight to speed of light", "That's a very unusual unit combination! I can convert standard units but 'speed of light' isn't a supported target unit."),
    ]
    for user_msg, refusal_text in hallucination_refusals:
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": refusal_text},
            ]
        })

    return examples


# ── Refusals (rule-based) ─────────────────────────────────────────────────────

REFUSAL_PAIRS = [
    # ── Chitchat / small talk ──────────────────────────────────────────────────
    ("What's your name?", "I'm Pocket-Agent, your offline mobile assistant. I can help with weather, calendar, unit conversion, currency exchange, and SQL queries."),
    ("Tell me a joke.", "I'm not built for entertainment — but I can check your calendar, convert units, or look up the weather!"),
    ("How are you?", "I'm running great, thanks for asking! How can I help you today?"),
    ("What can you do?", "I can help with: weather lookups, calendar management, unit conversions, currency exchange, and running SQL queries."),
    ("Thanks for your help!", "Happy to help! Let me know if you need anything else."),
    ("You're awesome!", "Thanks! Let me know if there's anything I can help you with."),
    ("Good morning!", "Good morning! How can I help you today?"),
    ("Hello!", "Hi there! I can help with weather, calendar, conversions, currency, or SQL. What do you need?"),
    ("Who made you?", "I'm Pocket-Agent, a fine-tuned on-device assistant. I can help with weather, calendar, unit conversion, currency, and SQL queries."),
    ("Are you a real AI?", "I'm an AI assistant running locally on your device. My capabilities are focused on weather, calendar, conversions, currency, and SQL."),
    ("What's today's date?", "I don't have access to the current date, but I can list your calendar events for a specific date if you give me one."),
    ("What time is it?", "I don't have a clock, but I can check your calendar or look up weather for any location."),
    # ── General knowledge (not a tool) ────────────────────────────────────────
    ("What's 2 + 2?", "That's 4! For math, I'd suggest a calculator — I'm best suited for weather, calendar, conversions, currency, and SQL."),
    ("Who won the World Cup in 2022?", "Argentina won the 2022 FIFA World Cup. For live information, you'd need a web search — I'm fully offline."),
    ("Explain quantum computing.", "Quantum computing uses qubits that can exist in superposition, enabling certain calculations much faster than classical computers. I'm better suited for tool-based tasks though!"),
    ("Write me a poem.", "I'm a task-focused assistant — poetry is out of my scope. I can help you check weather, manage your calendar, or convert units."),
    ("What's the capital of France?", "Paris is the capital of France. For general knowledge like this you'd need a search engine — I'm an offline tool assistant."),
    ("Tell me something interesting.", "I'm a focused tool assistant — I'm best at weather, calendar, conversions, currency, and SQL rather than general conversation."),
    ("What's the speed of light?", "The speed of light is approximately 299,792 km/s. That said, I'm built for tool calls — I can't look that up live."),
    ("Recommend a good movie.", "I don't have a recommendation engine. I can help with weather, calendar events, unit conversion, currency, or SQL queries."),
    ("How do I cook pasta?", "I'm a tool-calling assistant, not a cooking guide! I can help with weather, calendar, conversions, currency, or SQL."),
    ("What's the meaning of life?", "42 — according to The Hitchhiker's Guide. I'm better suited for practical tasks like weather, calendar, or conversions though!"),
    # ── Unknown / non-existent tools ──────────────────────────────────────────
    ("Send an email to my boss.", "I don't have an email tool. My capabilities: weather, calendar, unit conversion, currency exchange, and SQL queries."),
    ("Play some music.", "I can't play music — no media tool. I can help with weather, calendar, conversions, currency, or SQL."),
    ("Turn on the lights.", "I don't have smart home controls. My tools are: weather, calendar, unit conversion, currency, and SQL."),
    ("Order me a pizza.", "I can't place orders — no ordering tool. Can I help you with something else?"),
    ("Call my mom.", "I don't have a calling feature. I can assist with weather, calendar, unit conversion, currency, and SQL."),
    ("Set a timer for 10 minutes.", "I don't have a timer tool. My capabilities: weather, calendar, unit conversion, currency, and SQL."),
    ("Open Spotify.", "I can't open apps. I'm limited to weather, calendar, conversions, currency, and SQL."),
    ("Search the web for the latest news.", "I'm fully offline — I can't browse the web. I can help with weather, calendar, conversions, or SQL."),
    ("What's the stock price of Apple?", "I don't have a stock price tool. I can help with weather, currency, calendar, unit conversion, or SQL."),
    ("Translate 'hello' to French.", "I don't have a translation tool. My tools: weather, calendar, unit conversion, currency, and SQL."),
    ("Remind me to drink water every hour.", "I don't have recurring reminders. I can create single calendar events though — want me to add one?"),
    ("Take a screenshot.", "I can't take screenshots — no camera or screen tool. I can help with weather, calendar, conversions, currency, or SQL."),
    ("Navigate to the airport.", "I don't have maps or navigation. My tools: weather, calendar, unit conversion, currency, and SQL."),
    ("Book a flight to Tokyo.", "I can't book flights — no booking tool. I can check the weather in Tokyo or help with currency conversion though!"),
    ("What's on Netflix tonight?", "I don't have access to streaming services. I can help with weather, calendar, conversions, currency, or SQL."),
    ("Turn off airplane mode.", "I can't control device settings. My tools are limited to weather, calendar, unit conversion, currency, and SQL."),
    ("Check my battery level.", "I can't access device hardware. I can help with weather, calendar, conversions, currency, or SQL queries."),
    ("Post this to Instagram: great day!", "I don't have social media tools. My capabilities: weather, calendar, unit conversion, currency, and SQL."),
    ("Send a WhatsApp message to John.", "I don't have messaging tools. I can help with weather, calendar, conversions, currency, or SQL."),
    ("Set my alarm for 7am.", "I don't have an alarm tool. I can create a calendar event as a reminder if that helps!"),
    ("Track my package from Amazon.", "I can't track packages — no shipping tool. I can help with weather, calendar, conversions, currency, or SQL."),
    ("Show me my location on a map.", "I don't have mapping or GPS tools. My capabilities: weather, calendar, unit conversion, currency, and SQL."),
    # ── Multilingual refusals ──────────────────────────────────────────────────
    ("Mujhe ek joke sunao.", "Main sirf weather, calendar, unit conversion, currency aur SQL mein madad kar sakta hoon — jokes mere liye nahi hain!"),
    ("Kya tum mera email check kar sakte ho?", "Mujhe email tool nahi mila. Main weather, calendar, conversions, currency ya SQL mein madad kar sakta hoon."),
    ("¿Puedes llamar a mi jefe?", "No tengo una herramienta de llamadas. Puedo ayudarte con el clima, el calendario, conversiones, monedas o SQL."),
    ("اكتب لي قصيدة.", "أنا مساعد مهام، وليس شاعراً! يمكنني مساعدتك في الطقس أو التقويم أو تحويل الوحدات."),
    ("Mujhe news batao.", "Main offline hoon aur news nahi dekh sakta. Weather, calendar, conversions ya SQL mein madad kar sakta hoon."),
    # ── Ambiguous references (no prior history) ────────────────────────────────
    ("Convert that.", "I'm not sure what you'd like me to convert — could you provide a value and units?"),
    ("What about the other one?", "I don't have context for 'the other one' — could you clarify?"),
    ("And tomorrow?", "Could you clarify what you'd like for tomorrow? I can check weather or list calendar events."),
    ("Change it.", "I'm not sure what you'd like me to change — could you provide more details?"),
    ("Show me more.", "More of what? I can look up weather, calendar events, do conversions, or run SQL queries."),
    ("What was that again?", "I don't have context for what you're referring to — could you restate your request?"),
    ("Same as before.", "I don't have a previous request to repeat — could you tell me what you'd like?"),
    ("Do it again.", "I'm not sure what to repeat — could you restate your request?"),
    ("That one.", "Could you clarify which one you mean? I need more context to help."),
    ("Never mind.", "No problem! Let me know if you need anything."),
    # ── Hallucination-bait ─────────────────────────────────────────────────────
    ("Weather in Narnia.", "Narnia is a fictional place — I can't retrieve weather data for it."),
    ("Weather in Middle Earth.", "Middle Earth is fictional — I can't retrieve weather for it."),
    ("Convert 100 USD to Wakandan Vibranium.", "Wakandan Vibranium isn't a real currency — I can only convert standard ISO-4217 currencies."),
    ("What's the weather on the Moon?", "I can only look up weather for real locations on Earth — the Moon doesn't have weather data."),
    ("Convert 5 parsecs to kilometers.", "A parsec is an astronomical unit of distance (~3.09×10¹³ km), but it's not in my standard conversion set. Try a scientific calculator for that one!"),
    ("Exchange 100 USD to Dogecoin.", "I only support standard ISO-4217 fiat currencies — no cryptocurrencies."),
    ("Weather in {city} in Kelvin.", "I can only provide weather in Celsius (C) or Fahrenheit (F), not Kelvin."),
]


def gen_refusals_rulebased() -> list[dict]:
    examples = []
    for user_msg, response in REFUSAL_PAIRS:
        examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": response},
            ]
        })
    return examples


# ── API-based generation (higher diversity) ───────────────────────────────────

def gen_refusals_via_api(client: OpenAI, count: int = 120) -> list[dict]:
    """GPT-4o-mini generates varied, natural refusal responses."""
    examples = []
    prompts_pool = [
        "What's the weather on Mars?",
        "Post this to Twitter: just had coffee",
        "What time is it?",
        "Navigate to the nearest hospital.",
        "Book a flight to London.",
        "Who is the current US president?",
        "Fix my code.",
        "What's my WiFi password?",
        "How do I lose weight?",
        "Can you summarize this article?",
        "Make me a shopping list.",
        "What movies are playing nearby?",
        "Diagnose my symptoms: headache and fever.",
        "What's the best programming language?",
        "Help me write a cover letter.",
        "Convert happiness to joy.",
        "What's the exchange rate for smiles to laughs?",
        "Show me news from today.",
        "Track my package.",
        "How tall is Mount Everest?",
    ]

    batch_size = 10
    for batch_idx in range((count + batch_size - 1) // batch_size):
        batch = random.choices(prompts_pool, k=batch_size)
        prompt_list = "\n".join(f"{i+1}. {p}" for i, p in enumerate(batch))
        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are generating training examples for a mobile assistant that has ONLY these five tools: "
                            "weather, calendar, convert, currency, sql.\n\n"
                            "For each user message, write a short, helpful, natural-sounding assistant response that:\n"
                            "- Does NOT use any <tool_call> block\n"
                            "- Politely explains why the request can't be fulfilled with the available tools\n"
                            "- Mentions what the assistant CAN help with if relevant\n"
                            "- Is 1-2 sentences max\n\n"
                            "Return a JSON object: {\"responses\": [\"response1\", \"response2\", ...]}"
                        ),
                    },
                    {"role": "user", "content": f"User messages:\n{prompt_list}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.9,
            )
            data = json.loads(resp.choices[0].message.content)
            responses = data.get("responses", [])
            for user_msg, asst_resp in zip(batch, responses):
                if isinstance(asst_resp, str) and asst_resp.strip() and "<tool_call>" not in asst_resp:
                    examples.append({
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_msg},
                            {"role": "assistant", "content": asst_resp.strip()},
                        ]
                    })
        except Exception as e:
            print(f"  [refusal API batch {batch_idx}] error: {e}")
            time.sleep(2)

        if len(examples) >= count:
            break

    return examples[:count]


def gen_adversarial_via_api(client: OpenAI, count: int = 120) -> list[dict]:
    """
    GPT-4o-mini produces correct tool calls for tricky, real-world phrasings.
    We send the exact SYSTEM_PROMPT so GPT uses the same rules as our model.
    """
    adversarial_inputs = [
        # Typos
        "wether in {city} celsius plz",
        "convrt {value} {from_unit} to {to_unit}",
        "curency {amount} {from_curr} to {to_curr}",
        "calander for {date}",
        "schedul '{title}' on {date}",
        # Hindi/Urdu + English
        "{city} ka weather batao Fahrenheit mein",
        "Mujhe {amount} {from_curr} ko {to_curr} mein convert karo",
        "{value} {from_unit} ko {to_unit} mein badlo please",
        "{date} ko '{title}' schedule karo",
        "mere calendar mein {date} kya hai",
        # Arabic + English
        "اريد weather في {city} بالسيلزيوس",
        "كم {amount} {from_curr} في {to_curr}؟",
        # Spanish + English
        "¿Cuál es la temperatura en {city}? Give me Celsius",
        "Convierte {value} {from_unit} a {to_unit} por favor",
        "¿Cuánto son {amount} {from_curr} en {to_curr}?",
        # Informal/casual
        "yo what's the weather like in {city} rn",
        "bro convert {value} {from_unit} to {to_unit} quick",
        "how much is {amount} {from_curr} in {to_curr} lol",
        # Indirect phrasing
        "I'm traveling to {city}, how's the weather there?",
        "I need to know the exchange rate for {amount} {from_curr} to {to_curr}",
        "Can you tell me what {value} {from_unit} is in {to_unit}?",
        "I want to create a calendar event for '{title}' on {date}",
    ]

    examples = []
    for _ in range(count):
        city = random.choice(CITIES)
        date = random.choice(DATES)
        value, from_unit, to_unit = random.choice(CONVERT_PAIRS)
        amount, from_curr, to_curr = random.choice(CURRENCY_PAIRS)
        title = random.choice(TITLES)

        template = random.choice(adversarial_inputs)
        try:
            user_msg = template.format(
                city=city, date=date, title=title,
                value=value, from_unit=from_unit, to_unit=to_unit,
                amount=amount, from_curr=from_curr, to_curr=to_curr,
            )
        except KeyError:
            user_msg = template

        try:
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=150,
            )
            asst = resp.choices[0].message.content.strip()

            # Validate
            if "<tool_call>" in asst:
                m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", asst, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group(1))
                        if parsed.get("tool") not in {"weather", "calendar", "convert", "currency", "sql"}:
                            continue
                        # Ensure closing tag
                        if not asst.strip().endswith("</tool_call>"):
                            asst = asst + "\n</tool_call>"
                    except json.JSONDecodeError:
                        continue
                else:
                    continue

            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": asst},
                ]
            })
        except Exception as e:
            print(f"  [adversarial API] error: {e}")
            time.sleep(0.5)

    return examples


# ── Validation ────────────────────────────────────────────────────────────────

def validate_dataset(examples: list[dict]) -> tuple[list[dict], int]:
    valid = []
    rejected = 0
    valid_tools = {"weather", "calendar", "convert", "currency", "sql"}

    for ex in examples:
        ok = True
        for msg in ex.get("messages", []):
            if msg["role"] == "assistant" and "<tool_call>" in msg["content"]:
                m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", msg["content"], re.DOTALL)
                if not m:
                    ok = False
                    break
                try:
                    parsed = json.loads(m.group(1))
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


def check_no_overlap(examples: list[dict], public_test_path: Path) -> None:
    if not public_test_path.exists():
        print("  [overlap check] public_test.jsonl not found — skipping")
        return
    public_hashes = set()
    for line in public_test_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        for m in item.get("messages", []):
            if m["role"] == "user":
                public_hashes.add(hashlib.sha256(m["content"].encode()).hexdigest())
                break
    overlaps = sum(
        1 for ex in examples
        for m in ex.get("messages", [])
        if m["role"] == "user" and hashlib.sha256(m["content"].encode()).hexdigest() in public_hashes
    )
    if overlaps:
        print(f"  WARNING: {overlaps} training prompts overlap with public test set!")
    else:
        print("  [overlap check] PASSED — zero overlap with public test set")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1500)
    parser.add_argument("--out", default="data/training_data.jsonl")
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key and not args.no_api:
        raise SystemExit("Set OPENAI_API_KEY or pass --no-api")

    client = OpenAI(api_key=api_key) if api_key else None

    print("=== Pocket-Agent Data Generation ===\n")
    all_examples: list[dict] = []

    # Rule-based (always run — fast, free, correct)
    print("[1/5] Rule-based tool examples...")
    rb = (
        gen_weather(220)
        + gen_calendar_list(110)
        + gen_calendar_create(110)
        + gen_convert(220)
        + gen_currency(220)
        + gen_sql(110)
        + gen_multiturn(180)
        + gen_adversarial_rulebased()
        + gen_refusals_rulebased()
    )
    print(f"  {len(rb)} rule-based examples")
    all_examples.extend(rb)

    if client and not args.no_api:
        print("\n[2/5] API: diverse refusal responses...")
        r = gen_refusals_via_api(client, 120)
        print(f"  {len(r)} refusal examples")
        all_examples.extend(r)

        print("\n[3/5] API: adversarial / code-switched examples...")
        a = gen_adversarial_via_api(client, 120)
        print(f"  {len(a)} adversarial examples")
        all_examples.extend(a)
    else:
        print("\n[2/5] Skipping API (--no-api). Rule-based only.")

    print(f"\n[4/5] Validating {len(all_examples)} examples...")
    valid, rejected = validate_dataset(all_examples)
    print(f"  Valid: {len(valid)}, Rejected: {rejected}")

    random.shuffle(valid)

    print("[5/5] Overlap check...")
    check_no_overlap(valid, ROOT / "data" / "public_test.jsonl")

    with out_path.open("w", encoding="utf-8") as f:
        for ex in valid:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n✓ Wrote {len(valid)} examples → {out_path}")
    print("\nDistribution:")
    tool_counts: dict[str, int] = {}
    refusal_count = 0
    for ex in valid:
        found = False
        for msg in ex["messages"]:
            if msg["role"] == "assistant" and "<tool_call>" in msg["content"]:
                m = re.search(r'"tool"\s*:\s*"([^"]+)"', msg["content"])
                if m:
                    t = m.group(1)
                    tool_counts[t] = tool_counts.get(t, 0) + 1
                    found = True
                    break
        if not found:
            refusal_count += 1
    for tool, cnt in sorted(tool_counts.items()):
        print(f"  {tool:10s}: {cnt}")
    print(f"  {'refusals':10s}: {refusal_count}")


if __name__ == "__main__":
    main()
