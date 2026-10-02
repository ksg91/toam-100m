"""
Hand-written example tools and requests, used by examples/quickstart.py and as
a fallback prompt set for the release verification scripts. They were written
for this repository and are not taken from any training or evaluation set.
"""

TOOLS = [
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
            "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "send_email", "description": "Send an email.",
        "parameters": {"type": "object", "properties": {
            "to": {"type": "string", "format": "email"},
            "subject": {"type": "string"},
            "body": {"type": "string"}},
            "required": ["to", "subject", "body"]}}},
    {"type": "function", "function": {
        "name": "create_calendar_event", "description": "Create an event in the user's calendar.",
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"},
            "date": {"type": "string", "format": "date", "description": "YYYY-MM-DD"},
            "start_time": {"type": "string", "description": "HH:MM, 24-hour"},
            "duration_minutes": {"type": "integer", "default": 30}},
            "required": ["title", "date", "start_time"]}}},
    {"type": "function", "function": {
        "name": "convert_currency", "description": "Convert an amount from one currency to another.",
        "parameters": {"type": "object", "properties": {
            "amount": {"type": "number"},
            "from_currency": {"type": "string", "description": "ISO 4217 code, e.g. USD"},
            "to_currency": {"type": "string", "description": "ISO 4217 code, e.g. EUR"}},
            "required": ["amount", "from_currency", "to_currency"]}}},
    {"type": "function", "function": {
        "name": "delete_file", "description": "Permanently delete a file from the user's drive.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}},
            "required": ["path"]}}},
]

# (request, policy or None, state or None)
REQUESTS = [
    ("What's the weather in Pune right now?", None, None),
    ("How warm is it in Chicago? Fahrenheit please.", None, None),
    ("Weather in Paris and in Berlin, please.", None, None),
    ("Email priya@example.com with subject 'Lunch' and say see you at 1pm.", None, None),
    ("Send an email to my manager.", None, None),
    ("How much is 250 US dollars in euros?", None, None),
    ("Put 'Dentist' in my calendar tomorrow at 9:30.", None, {"today": "2026-10-02"}),
    ("Add a 1-hour sync with the design team on 2026-10-14 at 15:00.", None, None),
    ("Delete /reports/old_draft.docx", "Confirm with the user before deleting anything.", None),
    ("Book me a flight to Tokyo.", None, None),
    ("Who won the football world cup in 2010?", None, None),
]
