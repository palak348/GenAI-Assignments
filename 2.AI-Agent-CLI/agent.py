#!/usr/bin/env python3
"""
AI Agent CLI
A conversational terminal agent powered by Google Gemini that reasons
through tasks step-by-step and produces real output files.

Usage:
    python agent.py

Then type: Create a Scaler Academy website clone with header, hero, and footer
"""

import os
import json
import re
import webbrowser
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ── Groq setup ────────────────────────────────────────────────────────────────
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY:
    raise SystemExit(
        "\033[91mError: GROQ_API_KEY not found.\033[0m\n"
        "1. Get a free key at: https://console.groq.com\n"
        "2. Add to .env: GROQ_API_KEY=your_key_here"
    )

client = Groq(api_key=API_KEY)
MODEL  = "llama-3.1-8b-instant"

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a coding agent. Each reply must be exactly ONE valid JSON object — no markdown, no code fences, no extra text.

Step sequence: START -> THINK -> TOOL -> [OBSERVE from system] -> TOOL/OUTPUT

JSON schemas:
  {"step":"START|THINK|OUTPUT","content":"..."}
  {"step":"TOOL","content":"reason","tool_name":"name","tool_args":{...}}

Available tools:
  write_file      - args: {"filename":"output/index.html","content":"<full html>"}
  open_in_browser - args: {"filepath":"output/index.html"}

Rules (obey strictly):
  1. One JSON object per reply. Never combine steps.
  2. Wait for OBSERVE before the next step after every TOOL.
  3. ONLY write to output/index.html. NEVER create any other file (.css, .js, extra .html).
  4. output/index.html must have ALL CSS inside <style> and ALL JS inside <script> tags. Minimum 3000 characters.
  5. After at most 1 THINK step, call write_file with the complete HTML immediately.
  6. After write_file succeeds, call open_in_browser, then OUTPUT.
"""

# ── Tool implementations ───────────────────────────────────────────────────────
def write_file(args: dict) -> str:
    filename = args.get("filename", "")
    content  = args.get("content", "")
    if not filename:
        return "Error: 'filename' is required."
    if not content:
        return "Error: 'content' is empty."
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return (
        f"File '{path}' written successfully "
        f"({len(content):,} characters, {path.stat().st_size:,} bytes)."
    )

def open_in_browser(args: dict) -> str:
    filepath = args.get("filepath", "")
    if not filepath:
        return "Error: 'filepath' is required."
    path = Path(filepath).resolve()
    if not path.exists():
        return f"Error: '{path}' does not exist. Run write_file first."
    webbrowser.open(path.as_uri())
    return f"Opened '{path}' in the default browser."

TOOL_MAP = {
    "write_file":      write_file,
    "open_in_browser": open_in_browser,
}

# ── Terminal colors ───────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
BLUE   = "\033[94m"
YELLOW = "\033[93m"
PURPLE = "\033[95m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
RED    = "\033[91m"

STEP_COLOR = {
    "START":   BLUE,
    "THINK":   YELLOW,
    "TOOL":    PURPLE,
    "OBSERVE": CYAN,
    "OUTPUT":  GREEN,
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def print_step(parsed: dict) -> None:
    step  = parsed.get("step", "UNKNOWN")
    color = STEP_COLOR.get(step, "")
    label = f"{color}{BOLD}[{step}]{RESET}"
    if step == "TOOL":
        tool_name = parsed.get("tool_name", "?")
        preview   = json.dumps(parsed.get("tool_args", {}))
        if len(preview) > 100:
            preview = preview[:97] + "..."
        print(f"\n{label} {PURPLE}{tool_name}{RESET}  {DIM}{preview}{RESET}")
    else:
        content = parsed.get("content", "")
        if len(content) > 320:
            content = content[:317] + "..."
        print(f"\n{label} {content}")


def extract_json(text: str) -> dict:
    """Parse JSON from model output, stripping markdown code fences if present."""
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)


def normalize_tool_args(args) -> dict:
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            result = json.loads(args)
            return result if isinstance(result, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}

# ── Core agent loop ───────────────────────────────────────────────────────────
def call_model(history: list) -> str:
    """Send the current history to Groq and return the raw text response."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=history,
        max_tokens=768,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def add_user(history: list, text: str) -> None:
    history.append({"role": "user", "content": text})


def add_model(history: list, text: str) -> None:
    history.append({"role": "assistant", "content": text})


def run_agent_turn(user_input: str, history: list) -> None:
    """
    Process one user instruction through the full agent reasoning loop:
    START -> THINK... -> TOOL -> OBSERVE -> ... -> OUTPUT
    Mutates the shared history list so context persists across turns.
    """
    add_user(history, user_input)
    parse_errors  = 0
    think_count   = 0
    step_count    = 0
    MAX_STEPS     = 12
    empty_content = 0

    while True:
        step_count += 1
        if step_count > MAX_STEPS:
            print(f"\n{RED}{BOLD}[ABORT]{RESET} Reached {MAX_STEPS} steps — stopping loop.")
            break
        raw = call_model(history)

        # Parse JSON
        try:
            parsed = extract_json(raw)
            parse_errors = 0
        except (json.JSONDecodeError, ValueError):
            parse_errors += 1
            add_model(history, raw[:200])  # truncate bad responses to keep history small
            if parse_errors >= 3:
                print(f"\n{RED}{BOLD}[ERROR]{RESET} Model failed to return valid JSON 3 times. Aborting.")
                print(f"{DIM}Last response:\n{raw[:400]}{RESET}")
                break
            print(f"\n{RED}[PARSE ERROR]{RESET} Not valid JSON — retrying...")
            add_user(history, "Your response was not valid JSON. Reply with a single JSON object only — no markdown, no code fences.")
            continue

        # Record model response in history
        add_model(history, raw)

        # Display the step
        print_step(parsed)
        step = parsed.get("step", "")

        # Route by step
        if step in ("START", "THINK"):
            think_count += 1
            if think_count >= 2:
                add_user(history, "You have thought enough. Call write_file NOW with the complete output/index.html.")
            else:
                add_user(history, "Continue to the next step.")

        elif step == "TOOL":
            think_count = 0
            tool_name = parsed.get("tool_name", "")
            tool_args = normalize_tool_args(parsed.get("tool_args", {}))

            if tool_name not in TOOL_MAP:
                observe_content = (
                    f"Error: Tool '{tool_name}' does not exist. "
                    f"Available: {list(TOOL_MAP.keys())}"
                )
            else:
                try:
                    observe_content = TOOL_MAP[tool_name](tool_args)
                except Exception as exc:
                    observe_content = f"Error in '{tool_name}': {type(exc).__name__}: {exc}"

            print(f"\n{CYAN}{BOLD}[OBSERVE]{RESET} {observe_content}")

            # Abort if model repeatedly sends write_file with no content
            if observe_content == "Error: 'content' is empty.":
                empty_content += 1
                if empty_content >= 3:
                    print(f"\n{RED}{BOLD}[ABORT]{RESET} Model sent empty content 3 times in a row — stopping.")
                    break
            else:
                empty_content = 0

            # Replace the last assistant message (which has the full HTML in tool_args)
            # with a compact summary to keep history tokens small
            if tool_name == "write_file" and history and history[-1]["role"] == "assistant":
                history[-1] = {"role": "assistant", "content": json.dumps({
                    "step": "TOOL", "tool_name": tool_name,
                    "tool_args": {"filename": tool_args.get("filename","")}
                })}

            add_user(history, json.dumps({"step": "OBSERVE", "content": observe_content}))

        elif step == "OUTPUT":
            print(f"\n{DIM}{'─' * 60}{RESET}")
            break

        else:
            add_user(history, f"Unexpected step '{step}'. Follow: START -> THINK -> TOOL -> OUTPUT.")

# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    history = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Banner
    print(f"\n{BOLD}{'=' * 62}")
    print(f"   AI Agent CLI  |  Groq LLaMA 3   |  Website Builder")
    print(f"{'=' * 62}{RESET}")
    print(f"{DIM}Type your instruction below. Type 'exit' to quit.")
    print(f"Try: Create a Scaler Academy website clone{RESET}\n")

    # Interactive prompt loop
    while True:
        try:
            user_input = input(f"{BOLD}You:{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Goodbye!{RESET}")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q", "bye"):
            print(f"{DIM}Goodbye!{RESET}")
            break

        print()
        run_agent_turn(user_input, history)
        # Reset history to system prompt only — prevents token accumulation across turns
        history.clear()
        history.append({"role": "system", "content": SYSTEM_PROMPT})
        print()


if __name__ == "__main__":
    main()
