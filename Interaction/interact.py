import re
import ollama
from Perception.voice.speak import voice_output, play_sound
from Shared.utils.timing import timed
from Shared.Constants.sounds import tool_tone
from Shared.Tools.scout_tools import TOOL_SPECS
from Testing.preview.server import preview

# Helper functions
def load_system_prompt(path):
    with open(path, "r") as f:
        return f.read()


def _split_sentences(text: str) -> list[str]:
    """Split text on sentence-ending punctuation, keeping the punctuation."""
    parts = re.findall(r"[^.!?\n]+[.!?\n]+|[^.!?\n]+$", text)
    return [p.strip() for p in parts if p.strip()]


def run(conversation, system_prompt=None, model=None, tool_handler=None, max_iterations=5):
    """
    Run an agentic turn. The model may call tools across multiple iterations,
    interleaving speech and actions.
    
    Mutates `conversation` in place to record assistant turns and tool results.
    Yields spoken text chunks for the caller to track in spoken_words history.
    """
    full_messages = [{"role": "system", "content": system_prompt}] + conversation
    
    for iteration in range(max_iterations):
        stream = ollama.chat(
            model=model,
            messages=full_messages,
            tools=TOOL_SPECS,
            think=False,
            stream=True,
        )
        
        full_response = ""    # accumulates entire assistant text
        speech_buffer = ""    # accumulates until sentence boundary
        tool_calls = []       # collected from any chunk that has them
        
        for i, msg in enumerate(stream):
            chunk = msg["message"].get("content", "") or ""
            chunk_tools = msg["message"].get("tool_calls")
            
            if chunk_tools:
                tool_calls.extend(chunk_tools)
            
            if chunk:
                full_response += chunk
                speech_buffer += chunk
                
                # Flush completed sentences to TTS
                # Speak regardless if we don't yet know tools are coming
                if speech_buffer.endswith((".", "!", "?", "\n")):
                    sentences = _split_sentences(speech_buffer)
                    for s in sentences:
                        print(s)
                        voice_output(s)
                        preview.add_event("scout_speech", {"text": s})
                    speech_buffer = ""
        
        # Flush any remaining speech buffer (no tools coming)
        if not tool_calls and speech_buffer.strip():
            print(speech_buffer.strip())
            voice_output(speech_buffer.strip())
            preview.add_event("scout_speech", {"text": s})
        
        # Record assistant turn — once, after the full stream
        assistant_entry = {"role": "assistant", "content": full_response}
        if tool_calls:
            assistant_entry["tool_calls"] = tool_calls
        conversation.append(assistant_entry)
        full_messages.append(assistant_entry)
        
        # No tools? We're done.
        if not tool_calls:
            return
        
        # Execute tools, append results, loop back for next iteration
        play_sound(tool_tone())
        for call in tool_calls:
            fn = call["function"]
            name = fn["name"]
            args = fn.get("arguments", {}) or {}
            print(f"[Tools] Calling {name}({args})")
            
            with timed(f"tool_{name}"):
                if tool_handler is None:
                    result = "Error: no tool handler configured"
                else:
                    preview.add_event("tool_call", {"name": name, "args": args})
                    result = tool_handler.dispatch(name, args)
                    preview.add_event("tool_result", {"name": name, "result": result[:300]})
            
            print(f"[Tools] Result: {result[:100]}{'...' if len(result) > 100 else ''}")
            tool_entry = {"role": "tool", "content": result}
            conversation.append(tool_entry)
            full_messages.append(tool_entry)
    
    print(f"[Interact] Hit max_iterations ({max_iterations})")