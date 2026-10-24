# RepoSherlock

A terminal AI agent that investigates unfamiliar **Python** repositories. You ask a question; it
finds the relevant functions with Tree-sitter, builds a call graph, asks a **local** model
(Ollama) to explain them, and prints a report with **real file names and line numbers**.
Every investigation step is announced live through **DNotifier**.

* Read-only: it parses text, it never imports or runs the code it analyzes.
* Local AI only: inference runs in Ollama. DNotifier's hosted AI is **not** used.
* Honest: facts computed from the code are separated from the model's hypotheses, and
  DNotifier delivery is reported as *confirmed* only when it is actually confirmed.

## How DNotifier is used (and verified)

| Piece | What it does |
|---|---|
| `agent` client (`reposherlock-agent`) | Sends `investigation_started`, `repository_scanned`, `finding_discovered`, `report_completed`, `investigation_failed` over a DNotifier WebSocket (`DNotifier.send`). |
| `ack` client (`reposherlock-ack`) | A second DNotifier connection inside the same run. An event counts as **confirmed** only when this client receives it back through DNotifier with the same event id. (`send()` alone only returns an id; it is not proof of delivery.) |
| `monitor` client (`python main.py monitor`) | Run in a second terminal (or on a teammate's PC) to watch the investigation live. |

If DNotifier is unreachable, unconfigured or rejects the plan, the app says so, keeps working,
and writes the events to `reposherlock_events.local.jsonl`, a file clearly labelled
**LOCAL-ONLY FALLBACK**. It never prints success for an unconfirmed event.

Source: official docs at https://dnotifier.gitbook.io/product-docs (Python SDK `pip install dnotifier`).
All code lives in `notifier.py`.

## Setup (Windows, PowerShell)

```powershell
cd path\to\reposherlock
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry.

1. **Ollama**: install from https://ollama.com/download, then:
   ```powershell
   ollama pull qwen2.5-coder:3b
   ```
2. **DNotifier**: open https://app.dnotifier.com, open your *RepoSherlock* app, copy the **App ID**
   and **Secret** into `.env` (`DNOTIFIER_APP_ID`, `DNOTIFIER_SECRET`). Never commit `.env`.
3. **Check everything:**
   ```powershell
   python main.py doctor
   ```
   You want to see `Round trip verified`. Until you do, don't claim DNotifier delivery.

## Use

```powershell
python main.py analyze sample_project
python main.py ask sample_project "How does authentication work?"
python main.py ask sample_project "Which functions call login()?"
python main.py ask sample_project "Explain the execution flow from main() to connect_database()"
python main.py ask sample_project "Which functions might be responsible for KeyError: 'db_url'?" --save report.md
python main.py ask sample_project "Summarize the architecture of this repository"
python main.py functions sample_project
```

Useful flags for `ask`: `--save report.md`, `--no-ai` (static only), `--model`, `--timeout 400`,
`--show-code`, `--no-dnotifier` (labelled local-only run).

Live view in a second terminal: `python main.py monitor`.

## Tests

```powershell
python -m pytest -q
```

## 60-second demo script

1. *(0:00)* "Onboarding to an unfamiliar codebase is slow, and long-running AI analysis happens where nobody sees it. RepoSherlock fixes both."
2. *(0:10)* Terminal 2: `python main.py monitor`. "This is a teammate's screen. It is connected through DNotifier."
3. *(0:18)* Terminal 1: `python main.py ask sample_project "Which functions might be responsible for KeyError: 'db_url'?" --save report.md`
4. *(0:25)* Point at the monitor as events appear: `investigation_started`, `repository_scanned`, `finding_discovered`, `report_completed`. "Each event travels over DNotifier, and the agent checks it came back, so we can show `confirmed`, not just 'sent'."
5. *(0:40)* Scroll the report: "Facts computed from the code are separate from the model's hypotheses. Every function has a file and line number."
6. *(0:50)* "The AI runs locally in Ollama, so no code leaves the machine. Only small status events go through DNotifier. Without DNotifier the investigation is invisible; with it, the whole team follows along in real time."

## Troubleshooting

| Problem | Fix |
|---|---|
| `python` not found | Install Python 3.10+ from python.org and tick "Add to PATH". Or use `py -3`. |
| Cannot activate venv | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `Ollama is not running` | Start the Ollama app, or run `ollama serve` in another terminal. |
| `Model ... is not installed` | `ollama pull qwen2.5-coder:3b` |
| Model times out | `--timeout 400`; close other heavy apps; first run loads the model into RAM. |
| `Authentication failed` | Re-copy App ID / Secret from the dashboard into `.env` (no quotes, no spaces). |
| Events show `SENT, NOT CONFIRMED` | Run `python main.py doctor`. Check plan limits (3 user ids are used), and that the secret belongs to this app. |
| `tree-sitter` install fails | `python -m pip install --upgrade pip`, then reinstall. Needs Python 3.10+ (64-bit). |
| Garbled characters | Use Windows Terminal, or `chcp 65001`. |

## Limits (be upfront in your demo)

* Python only. The call graph matches calls **by name** (static), so dynamic dispatch, callbacks and
  framework hooks are invisible and a few edges can be ambiguous (the report flags them).
* A 3B model can be wrong. The report cites only snippet ids, and any file path or line number the model
  invents is flagged automatically, but the explanation is still a lead, not proof.
* It does not detect every bug and it never reproduces failures, because nothing is executed.
