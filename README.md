# LangChain Agent Evaluation System

## Objective

This project does two things:

1. **Builds an AI agent** that handles customer refund requests. It looks up the
   customer's purchases, checks the refund rules, works out the amount and issues a
   refund ticket.
2. **Evaluates that agent automatically.** A set of test questions with known correct
   answers is run through the agent, and a second AI model (the "judge") scores how well
   the agent did.

The goal is to measure whether the agent makes the right decision, calls the right
tools with the right arguments, and follows a sensible plan, using numbers instead of
manual checking.

## Refund rules the agent must follow

1. No refund more than 30 days after purchase, and no purchase can be dated in the future.
2. Identify the customer first, then look up their purchase history.
3. Refund only the items the customer asks for.
4. Refund only items the customer actually bought.
5. Tell the customer either that a refund ticket was issued (with the amount) or why no
   refund was given.
6. If the username isn't found, say it's invalid and ask for a valid one.

These rules are written in [agent/prompts.py](agent/prompts.py).

---

## Project structure

```
LangChain_Agent_Evaluation_System/
├── .env                      # API keys (GROQ_API_KEY)
├── requirements.txt          # Python packages
├── agent/                    # Part 1: the refund agent
│   ├── agent.py              #   ▶ RUN THIS to chat with the agent
│   ├── models.py             #   which LLM the agent uses (Groq)
│   ├── prompts.py            #   system prompt with the refund rules
│   ├── tools.py              #   the 4 tools the agent can call
│   ├── data.py               #   fake customer purchase database
│   └── objects.py            #   output format (Answer) and PurchaseItem
└── evaluations/              # Part 2: testing the agent
    ├── eval.py               #   ▶ RUN THIS for the full evaluation (7 metrics)
    ├── dataset.py            #   7 test questions with expected answers
    └── model.py              #   the judge LLM (local Ollama gpt-oss:20b)
```

---

## Step 1: Setup (one time)

```bash
cd /home/roubish/Documents/LangChain_Agent_Evaluation_System
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Groq key (free at
https://console.groq.com/keys):

```
GROQ_API_KEY=gsk_...
```

For the judge, install [Ollama](https://ollama.com/download) and download the model
(about 13 GB):

```bash
ollama pull gpt-oss:20b
```

---

## Step 2: The agent (`agent/` folder)

### How to run it

```bash
cd agent
python agent.py
```

It starts with the message *"I would like to request a refund"* and then lets you chat.
Type your replies at the `reply:` prompt, for example `I am user123, refund the book`.
It stops when the agent has made its refund decision.

### What happens inside, in order

```
agent.py
  │
  ├─ 1. models.py   → loads the LLM: openai/gpt-oss-120b on Groq
  ├─ 2. prompts.py  → loads the system prompt (the refund rules)
  ├─ 3. tools.py    → loads the 4 tools (they read data.py)
  ├─ 4. objects.py  → loads the Answer format the agent must reply in
  │
  └─ 5. create_agent(...) builds the agent, then run_conversation() starts the chat loop:

        user message
            │
            ▼
        ┌─────────────── agent loop ───────────────┐
        │  LLM reads the rules + conversation      │
        │  and decides which tool to call:         │
        │                                          │
        │  get_user_purchase_history(user)         │ ← who is the customer, what did they buy
        │  get_today_date()                        │ ← to check the 30-day rule
        │  calculate_right_refund(user, items)     │ ← adds up the prices
        │  issue_refund_ticket(user, amount)       │ ← issues the refund
        │                                          │
        │  repeats until it has its final answer   │
        └──────────────────────────────────────────┘
            │
            ▼
        answer tool → Answer(answer="...", conversation=True/False)
            │
            ├─ conversation=True  → print the answer, ask you for a reply, loop again
            └─ conversation=False → print the answer, "the job is done!"
```

### The files explained

| File | What it does |
|---|---|
| [agent/models.py](agent/models.py) | Creates the LLM. The default is `openai/gpt-oss-120b` on Groq; set the `GROQ_AGENT_MODEL` environment variable to change it. |
| [agent/prompts.py](agent/prompts.py) | The system prompt: the refund rules, plus "always reply through the `answer` tool". |
| [agent/tools.py](agent/tools.py) | The 4 tools. They are plain Python functions marked with `@tool`. |
| [agent/data.py](agent/data.py) | Fake database: users `user123` to `user132`, each with items, prices and purchase dates. |
| [agent/objects.py](agent/objects.py) | `Answer`: the agent's reply text, plus `conversation`, which says whether the chat should continue. `PurchaseItem`: one item to refund. |
| [agent/agent.py](agent/agent.py) | Builds everything into one agent and runs the chat loop. |

`agent.py` uses two pieces of LangChain machinery:

- **`ToolStrategy(Answer)`** forces the agent's final reply to come back as a call to a
  tool named `answer`, so the program always gets structured data back instead of
  free text.
- **`ModelRetryMiddleware`** retries the LLM call if Groq rejects a turn because the model
  replied in plain text instead of calling a tool.

---

## Step 3: The evaluation (`evaluations/` folder)

### How to run it

```bash
cd evaluations
python eval.py
```

It takes a few minutes. Groq runs the agent and Ollama on your GPU runs the judge.

### What happens inside, in order

```
eval.py
  │
  ├─ 1. dataset.py → loads 7 test questions ("goldens"), each with:
  │                   • input            – what the customer says
  │                   • expected_output  – what a correct answer looks like
  │                   • expected_tools   – which tools should be called, with which arguments
  │
  ├─ 2. model.py   → loads the judge: gpt-oss:20b running locally on Ollama
  │
  ├─ 3. imports the agent from agent/agent.py (the chat loop does NOT start)
  │
  ├─ 4. PART A, for each of the 7 questions:
  │        • run the agent on the question
  │        • deepeval's CallbackHandler records every step (a "trace")
  │        • compact_trace() shrinks the trace so it is smaller for the judge
  │        • the judge scores the trace with the 4 trace metrics
  │        • save the final answer and the tools called as a test case
  │
  ├─ 5. PART B: the judge scores all 7 saved test cases with the other 3 metrics
  │
  └─ 6. prints a results table for each metric, then a pass rate
```

### The 7 metrics

| Metric | Question it answers | Pass if |
|---|---|---|
| **Task Completion** | Did the agent finish the refund task correctly? | ≥ 0.8 |
| **Step Efficiency** | Did it avoid unnecessary steps? | ≥ 0.8 |
| **Plan Quality** | Was its plan sensible? | ≥ 0.8 |
| **Plan Adherence** | Did it follow its own plan? | ≥ 0.8 |
| **Argument Correctness** | Did it pass the right arguments to the tools? | score only, no pass/fail |
| **Tool Correctness** | Did it call the expected tools with the expected arguments? | ≥ 0.9 |
| **Answer Correctness** (GEval) | Does its final answer reach the same conclusion as the expected answer? | ≥ 0.8 |

The first 4 read the full trace (Part A). The last 3 compare the agent's output with the
golden (Part B).

### The 7 test questions

| # | Customer says | Correct result |
|---|---|---|
| 1 | user123, refund book and pen | Ticket for $30 |
| 2 | user123, refund book | Ticket for $23 |
| 3 | user123, just show my purchases | No ticket, list the items |
| 4 | user999 (doesn't exist), refund book | Say the username is invalid and ask again |
| 5 | user126, refund water_bottle | No refund, older than 30 days |
| 6 | user126, refund backpack and water_bottle | Ticket for $45 (backpack only) |
| 7 | user126, refund sneakers | No refund, item never purchased |

---

## Full order of things to do

```
1. Setup        →  venv, pip install, .env with GROQ_API_KEY, ollama pull gpt-oss:20b
2. Try agent    →  cd agent && python agent.py          (chat with it by hand)
3. Evaluate     →  cd evaluations && python eval.py
4. Improve      →  change prompts.py or tools.py, run step 3 again, compare scores
```

---

## Things to know

- **Groq free tier limits.** The agent's model allows 8,000 tokens per minute and
  200,000 tokens per day. One full `eval.py` run uses roughly 100,000 tokens of agent
  calls. If you see `429 ... tokens per day`, wait for the allowance to recover or set
  `GROQ_AGENT_MODEL` to another model.
- **Purchase dates are real dates.** [agent/data.py](agent/data.py) uses fixed dates and
  `get_today_date()` returns the actual date today, so the 30-day rule depends on when you
  run the tests. Over time, items that are refundable now will go past 30 days and the
  expected answers in [evaluations/dataset.py](evaluations/dataset.py) will become wrong.
  Update the dates in `data.py` when that happens.
- **Switching the judge.** [evaluations/model.py](evaluations/model.py) holds the judge.
  The commented-out `GroqJudge` class in that file is the Groq version if you ever want to
  go back.
