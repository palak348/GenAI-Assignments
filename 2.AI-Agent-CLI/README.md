# AI Agent CLI — Website Builder

A conversational terminal agent powered by **Groq LLaMA 3** that reasons through tasks step-by-step and produces real output files.

## What It Does

Type a natural language instruction in the terminal. The agent thinks through the task, calls tools, observes results, and produces a working HTML website — all visible in real time.

```
You: Create a Scaler Academy website clone with header, hero section, and footer

[START]   Planning the website structure...
[THINK]   Will write a single index.html with inline CSS and JS...
[TOOL]    write_file → output/index.html
[OBSERVE] File written successfully (1,382 characters)
[TOOL]    open_in_browser → output/index.html
[OBSERVE] Opened in default browser
[OUTPUT]  Done!
```

## Agent Reasoning Loop

```
START → THINK → TOOL → OBSERVE → TOOL → OUTPUT
```

Each model reply is one JSON step. The agent never skips steps and always waits for tool results before continuing.

## Setup

**1. Clone the repo**
```bash
git clone <your-repo-url>
cd 2.AI-Agent-CLI
```

**2. Create a virtual environment**
```bash
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Add your API key**
```bash
cp .env.example .env
# Edit .env and paste your Groq API key
# Get a free key at: https://console.groq.com
```

**5. Run the agent**
```bash
python agent.py
```

## Usage

```
You: Create a Scaler Academy website clone with header, hero section, and footer
```

The agent will:
- Reason through the task step-by-step
- Write `output/index.html` with all CSS and JS inline
- Open the result in your default browser

Type `exit` to quit.

## Tools Available

| Tool | Description |
|---|---|
| `write_file` | Writes content to a file on disk |
| `open_in_browser` | Opens a local HTML file in the default browser |

## Project Structure

```
├── agent.py          # Main agent — reasoning loop + tool execution
├── requirements.txt  # Dependencies (groq, python-dotenv)
├── .env.example      # API key template
├── .env              # Your actual key (not committed)
└── output/
    └── index.html    # Generated Scaler Academy website clone
```

## Tech Stack

- **LLM**: Groq `llama-3.1-8b-instant` (free tier)
- **Python**: `groq`, `python-dotenv`, standard library only
- **Output**: Single self-contained HTML file with inline CSS + JS

## Sample Output

The agent generates a Scaler Academy website clone featuring:
- Sticky glassmorphism header with dropdown navigation
- Hero section with gradient title and stats
- Program cards with hover animations
- Company marquee, stats band, testimonials
- Footer with social links

---

*Built for Scaler Academy — GenAI Assignment 2*
