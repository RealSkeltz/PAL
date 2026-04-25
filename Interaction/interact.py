import re
import ollama
from Perception.voice.speak import voice_output, play_sound
from Shared.utils.timing import timed
from Shared.Constants.sounds import tool_tone
from Shared.Tools.scout_tools import TOOL_SPECS


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
        with timed(f"llm_call_{iteration}"):
            response = ollama.chat(
                model=model,
                messages=full_messages,
                tools=TOOL_SPECS,
                think=False,
            )
        msg = response["message"]
        text_content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls", [])
        
        # Speak any text the model produced this iteration
        if text_content.strip():
            for sentence in _split_sentences(text_content):
                yield sentence
                voice_output(sentence)
        
        # Record assistant turn (text + any tool calls) in conversation
        assistant_entry = {"role": "assistant", "content": text_content}
        if tool_calls:
            assistant_entry["tool_calls"] = tool_calls
        conversation.append(assistant_entry)
        full_messages.append(assistant_entry)
        
        # If no tools were called, the turn is done
        if not tool_calls:
            return
        
        # Execute each tool, append results to context for the next iteration
        for call in tool_calls:
            fn = call["function"]
            name = fn["name"]
            args = fn.get("arguments", {}) or {}
            print(f"[Tools] Calling {name}({args})")

            play_sound(tool_tone())
            print(f"[Tools] → {name}({args})")
            
            with timed(f"tool_{name}"):
                if tool_handler is None:
                    result = "Error: no tool handler configured"
                else:
                    result = tool_handler.dispatch(name, args)
            
            print(f"[Tools] Result: {result[:100]}{'...' if len(result) > 100 else ''}")
            tool_entry = {"role": "tool", "content": result}
            conversation.append(tool_entry)
            full_messages.append(tool_entry)
    
    # Reached max iterations without the model finishing
    print(f"[Tools] Max iterations ({max_iterations}) reached, stopping turn")
    fallback = "I got stuck working on that, let me know if you want me to try again."
    yield fallback
    voice_output(fallback)
    conversation.append({"role": "assistant", "content": fallback})