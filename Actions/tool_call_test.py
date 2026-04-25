import ollama
import json

MODEL = "qwen3.5:9b"

tools = [
    {
        "type": "function",
        "function": {
            "name": "save_note",
            "description": "Save a note for the user to refer back to later. Use when the user says to remember something, write something down, or make a note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The text content of the note",
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for current information. Use when the user asks about recent events, current facts, or anything you don't know.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query, a few words long",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_last_image",
            "description": "Save the most recently captured image to disk for later reference. Use when the user wants to keep, save, or remember what they just showed you.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Optional descriptive filename without extension. If omitted, a timestamp is used.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_timer",
            "description": "Set a timer that will alert the user after a duration. Use when the user asks to be reminded after some time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": "Duration in seconds",
                    },
                    "label": {
                        "type": "string",
                        "description": "Short description of what the timer is for",
                    },
                },
                "required": ["seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": "Retrieve previously saved notes. Use when the user asks what they noted, what they remember, or to recall something they saved.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
]

system_prompt = """You are Scout, a hands-free voice assistant for professionals working with their hands occupied. You have tools available to help the user manage information during their work. Use tools when the user clearly asks for the action they enable. For ordinary conversation, questions you can answer directly, or commentary on what you see, just respond normally without calling tools."""

test_cases = [
    # Should call save_note
    ("Make a note that the second valve was leaking.", "save_note"),
    ("Remember I parked on level 3.", "save_note"),
    
    # Should call search_web
    ("What's the current weather in Amsterdam?", "search_web"),
    ("Has the new iPhone been announced yet?", "search_web"),
    
    # Should call save_last_image
    ("Save that image, I want to look at it later.", "save_last_image"),
    ("Keep this picture as evidence.", "save_last_image"),
    
    # Should call set_timer
    ("Remind me in 10 minutes to check the oven.", "set_timer"),
    ("Tell me in half an hour to take a break.", "set_timer"),
    
    # Should call list_notes
    ("What did I note down earlier?", "list_notes"),
    ("Read me my notes.", "list_notes"),
    
    # Should NOT call any tool
    ("How does a transformer work?", None),
    ("This is going well.", None),
    ("Can you see what I'm holding?", None),
    
    # Tricky: phrased as a question but really a save request
    ("Could you note that the inspection passed?", "save_note"),
    
    # Tricky: search-y phrasing but answerable from training
    ("What's the capital of France?", None),
    
    # Tricky: ambiguous between search and just answering
    ("How tall is Mount Everest?", None),
]

for prompt, expected in test_cases:
    print(f"\n{'='*60}")
    print(f"USER: {prompt}")
    print(f"EXPECTED: {expected if expected else 'no tool call'}")
    
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        tools=tools,
    )
    msg = response["message"]
    
    called = []
    if msg.get("tool_calls"):
        for call in msg["tool_calls"]:
            fn = call["function"]
            args = fn.get("arguments", {})
            print(f"TOOL: {fn['name']}({json.dumps(args)})")
            called.append(fn["name"])
    
    if msg.get("content"):
        print(f"TEXT: {msg['content']}")
    
    # Quick pass/fail
    if expected is None:
        result = "PASS" if not called else f"FAIL (called {called})"
    else:
        result = "PASS" if expected in called else f"FAIL (called {called or 'nothing'})"
    print(f"RESULT: {result}")