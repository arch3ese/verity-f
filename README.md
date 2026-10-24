# Verity: LLM Judge Debiasing, Rubric Alignment & Dataset Exporter

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.3+-green.svg)](https://www.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **A research and evaluation studio for LLM-as-a-judge debiasing, programmatic constraint verification, human annotator calibration, and DPO training dataset export.**

---

## ⚠️ Security Notice & Operational Scope

**No Authentication:** Verity Studio and its FastAPI backend are designed for local evaluation, CI testing, and offline researcher workflows. The server includes **no built-in user authentication, role-based access control, or rate limiting**. Do not expose this server to the public Internet without an authenticating reverse proxy (e.g., NGINX, Cloudflare Access) and TLS encryption.

**Judge Accuracy Disclaimer:** Verity's judge and debiasing engine have **not been benchmarked against standard public leaderboards** (such as MT-Bench, AlpacaEval, or LMSYS Chatbot Arena). The framework implements defensive mechanisms against known judge pathologies (positional preference, verbosity inflation, negative constraint tokenization blindness), but real-world accuracy depends heavily on the underlying model and task domain.

---

## 🎯 What Verity Solves

LLM-as-a-judge pipelines are notoriously vulnerable to subtle biases:
1. **Positional Bias (First-Position Effect):** Judges systematically favor whichever candidate is presented first in the prompt.
2. **Verbosity Bias:** Judges disproportionately award higher scores to longer, wordier responses even when they contain fluff or hallucinations.
3. **Tokenization Blindness:** LLMs struggle to enforce negative constraints (e.g. *"do not use the letter 'e'"*) or count-based formatting because BPE tokenizers split words into multi-letter subwords.
4. **Style Hallucination:** Eloquent phrasing frequently tricks judges into overlooking factual inaccuracies.

Verity combines **programmatic neuro-symbolic verification** (regex, lipogram analysis, length tracking) with **dual-order presentation swapping** ($A \text{ vs } B$ and $B \text{ vs } A$) and **hierarchical 5-tier rubrics** to detect and flag these failure modes.

---

## 🏛️ System Architecture

```mermaid
graph TD
    subgraph Input Prompts & Candidates
        P[Prompt + Constraints] --> A[Candidate Response A]
        P --> B[Candidate Response B]
    end

    subgraph Deterministic Rule Verification
        A & B --> RV[RuleVerifier: Regex & Telemetry]
        RV --> TEL[Constraint Violations & Noise Detection]
    end

    subgraph Dual-Order Debiasing Judge
        A & B & TEL --> FWD[Forward Pass: A vs B]
        A & B & TEL --> REV[Reverse Pass: B vs A]
        FWD & REV --> CONS[Consensus & Drift Engine]
    end

    subgraph Calibration & Veto Guardrails
        CONS --> VETO[Tier 1 Safety & Tier 2 Hard Negative Veto]
        CONS --> CAL[In-Context GuidelineBank Calibration]
        VETO --> RES[Final Consensus: Model A / Model B / Tie / Indeterminate]
    end

    subgraph Downstream Artifacts
        RES --> DPO[DPO Preference Triplets JSONL]
        RES --> STATS[Cohen's Kappa vs Human Ground Truth]
```

---

## ✨ Core Components

### 1. Positional Debiasing Engine (`DebiasingJudge`)
Executes evaluations in both forward ($A \text{ vs } B$) and reverse ($B \text{ vs } A$) presentation orders concurrently:
- If the model's preference flips purely due to presentation order, the evaluation is marked with `positional_bias_detected = True` and downgraded to `INDETERMINATE`.
- If mean absolute rubric score drift exceeds `POSITIONAL_SCORE_DRIFT_THRESHOLD` (0.8 on 1–5 scale), positional drift is flagged even if the categorical winner did not invert.

### 2. Programmatic Constraint Verification (`RuleVerifier`)
Deterministic Python telemetry analyzes candidate text prior to LLM invocation:
- Forbidden letter/word counts (case-insensitive with word-boundary awareness).
- Sentence count, word count, character count.
- Non-lexical noise and semantic void detection.
- Telemetry is injected into the LLM judge prompt, preventing tokenizer hallucination.

### 3. Tiered Rubric Priority & Critical Veto
Evaluates candidates across 5 standardized dimensions:
1. **Tier 1 (Safety & Harm):** Weight 5.0. Actionable harm triggers an absolute veto disqualifying the candidate regardless of formatting.
2. **Tier 2 (Instruction Following & Negative Constraints):** Weight 4.0. Core adherence and negative rule compliance.
3. **Tier 3 (Factuality):** Weight 3.0. Grounded truth outranks style.
4. **Tier 4 (Reasoning & Logic):** Weight 2.0. Deductive coherence.
5. **Tier 5 (Style, Tone & Conciseness):** Weight 0.5. Secondary tiebreaker only.

### 4. Statistical Agreement Metrics (`calculate_cohens_kappa`)
- **Unweighted Cohen's Kappa:** Categorical agreement for winner labels.
- **Weighted Cohen's Kappa:** Linear and Quadratic weights for ordinal rubric scores, mathematically validated against `scikit-learn`.
- **Zero-Variance Handling:** Returns `NaN` with `"Undefined: Zero variance across raters"` rather than an incorrect 1.0.
- **Sample Size Warning:** Emits an explicit low-sample warning when $N < 10$ (`MIN_SAMPLE_SIZE`).

### 5. Atomic In-Context Calibration (`GuidelineBank`)
- Stores gold-standard human annotator overrides and rationale.
- Thread-safe persistence uses file-level locking (`threading.RLock`) and atomic writes (`tempfile` + `os.replace`) to eliminate race conditions and partial-write corruption.

---

## 📡 REST API Reference

Verity Studio runs as a FastAPI service on port 8000:

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive macOS-styled dark Studio UI. |
| `GET` | `/api/status` | Returns availability of configured model providers (Gemini, OpenAI, Ollama). |
| `GET` | `/api/presets` | Returns bundled benchmark evaluation prompt pairs from `sample_prompts.json`. |
| `GET` | `/api/rubrics` | Returns the 5 evaluation criteria, priority weights, and error taxonomy. |
| `GET` | `/api/guidelines` | Returns currently active human alignment calibration exemplars. |
| `POST` | `/api/evaluate` | Executes a debiased dual-order evaluation between Candidate A and Candidate B. |
| `POST` | `/api/generate-candidate` | Generates a completion using a selected LLM provider. |
| `POST` | `/api/vote` | Records a quick human vote and returns updated agreement rate. |
| `POST` | `/api/audit` | Submits a senior human rubric audit, appends calibration guideline, and exports DPO triplet. |
| `POST` | `/api/agreement` | Computes live Cohen's Kappa score between annotator votes and AI judge votes. |
| `POST` | `/api/benchmark/run` | Runs the automated debiased benchmark suite across all scenarios and returns agreement metrics. |

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/arch3ese/verity.git
cd verity

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install package and dependencies
pip install -r requirements.txt
pip install -e .
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```env
# Cloud Providers
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Local Provider
OLLAMA_BASE_URL=http://localhost:11434

# Verity Configuration (Optional)
VERITY_DEFAULT_MODEL=gemini-3.8-flash
VERITY_FALLBACK_MODELS=gemini-flash-latest,gemini-3.5-flash-lite
VERITY_TEMPERATURE=0.0
VERITY_MAX_PROMPT_LENGTH=50000
VERITY_MAX_RESPONSE_LENGTH=100000
VERITY_CORS_ORIGINS=http://localhost:8000,http://127.0.0.1:8000
```
*(Legacy `ALIGN_*` environment variables are also automatically supported for backwards compatibility).*

### 3. Launch Web Studio
```bash
verity ui
```
Open [http://localhost:8000](http://localhost:8000) in your browser.

---

## 🛠️ CLI Reference

### View Rubrics and Taxonomy
```bash
verity rubrics
```

### Run an Instant Pairwise Audit
```bash
verity audit \
  --prompt "What is 13 * 17?" \
  --model-a "13 * 17 = 221" \
  --model-b "13 * 17 = 231" \
  --provider mock \
  --model mock-judge
```

### Batch Benchmark Execution & DPO Export
```bash
verity benchmark \
  --dataset examples/sample_prompts.json \
  --provider mock \
  --model mock-judge \
  --export-dpo data/exports/dpo_curated.jsonl \
  --export-summary eval_results/summary.json
```

### Run Scaffold Evaluation Harness
```bash
python scripts/eval_harness.py --csv data/eval_placeholder.csv --provider mock
```

### Export Dataset (Experimental)
```bash
# Note: Feature is currently under development (see issue #12). Use 'verity benchmark --export-dpo' in the meantime.
verity export --dataset eval_results/summary.json --output data/exports/dpo.csv
```

---

## 🧪 Test Suite

Run the full automated test suite (including unit tests, concurrency tests, and Playwright browser tests):
```bash
pytest -v
```

Test coverage includes 96 passing tests:
- **`test_agreement.py` & `test_agreement_stat_fixes.py`:** Unweighted, linear, and quadratic Cohen's Kappa, zero-variance handling, sample size warnings.
- **`test_factory_and_models.py`:** Model ID configurability, temperature parameter forwarding, 404 error handling.
- **`test_guidelines_concurrency.py`:** 20-thread concurrent writes with atomic temp-file replace and zero data loss.
- **`test_api_hardening.py`:** Pydantic length bounds, sanitized 500 error handlers, explicit CORS origin validation.
- **`test_ui_polish.py`:** SVG checkmarks, ARIA popover accessibility, and keyboard navigation.
- **`test_parser_robustness.py`:** Markdown code fences, trailing text, refusals, empty bodies, and truncation.
- **`test_debiaser.py`:** Order invariance, positional bias detection, verbosity penalties.
- **`test_rule_verifier.py` & `test_rule_verifier_hardening.py`:** Deterministic constraint and negative rule verification.
- **`test_prompt_injection.py`:** Candidate tag isolation and header escaping.

---

## 📂 Project Structure

```text
verity/
├── pyproject.toml              # Package build configuration & metadata
├── requirements.txt            # Python dependencies
├── README.md                   # Documentation, API reference, & known limitations
├── .env.example                # Environment variable configuration template
├── src/verity/
│   ├── alignment/              # GuidelineBank (few-shot exemplars & atomic writes)
│   ├── exporters/              # DPO triplet export for fine-tuning
│   ├── judges/                 # Pairwise judge, RuleVerifier, & DebiasingJudge
│   ├── models/                 # Unified model factory (Gemini, OpenAI, Ollama, Mock)
│   ├── rubrics/                # 5-tier evaluation dimensions & priority hierarchy
│   ├── schemas/                # Pydantic domain models
│   ├── stats/                  # Cohen's Kappa (unweighted, linear, quadratic) & IAA
│   ├── web/                    # FastAPI server & static web studio interface
│   ├── runner.py               # Batch evaluation runner
│   └── cli.py                  # Typer CLI commands (ui, rubrics, audit, benchmark)
├── scripts/
│   ├── list_models.py          # Google GenAI model discovery & ID resolution audit
│   └── eval_harness.py         # CSV evaluation harness scaffold
├── tests/                      # Full pytest test suite (96 tests)
├── data/                       # Guidelines & export files
└── examples/                   # Sample prompts & demo script
```

---

## 🔍 Known Limitations

1. **Heuristic Rubric Weights:** The 5-tier priority weights (`Safety=5.0`, `Instruction=4.0`, `Factuality=3.0`, `Logic=2.0`, `Style=0.5`) reflect curated human guidelines rather than empirical loss gradients.
2. **Exact Override Matching:** Human annotator overrides in `GuidelineBank` match prompts via normalized string equality rather than dense semantic embeddings.
3. **Single-Node File Locking:** `GuidelineBank` persistence relies on in-process `threading.RLock` and local filesystem `os.replace`. It is not suitable for horizontally scaled multi-worker deployments without an external SQL database.
4. **Judge Non-Determinism:** While setting `temperature=0.0` produces consistent results on standard chat models, reasoning models (e.g. models with chain-of-thought) may exhibit minor stochastic variation across identical prompts.
5. **Payload Bounds:** Prompts are restricted to 50,000 characters and completions to 100,000 characters to prevent memory exhaustion.

---

---

## 🗺️ Roadmap & Open Items (TODO)

The following areas are partially implemented or currently tracked for upcoming milestones:
- [ ] **Multi-Turn Context & Dialogues:** Current debiasing pipeline assumes single-turn prompt-response pairs. Conversation history parsing is stubbed.
- [ ] **Async Batch Runner (`run_benchmark_async`):** Add semaphore-bounded concurrent evaluation to `runner.py`.
- [ ] **Vector Semantic Retrieval:** Replace naive substring matching in `GuidelineBank.search_semantic()` with dense vector embeddings.
- [ ] **Direct Hugging Face Hub Exporter:** Implement `export_huggingface()` in `dpo_exporter.py` using PyArrow.
- [ ] **Multi-Rater Fleiss' Kappa:** Complete formulation in `agreement.py` for arbitrary rater pools.
- [ ] **Custom Regex Validation Hardening:** Add timeout bounds and syntax error catching to user-supplied regexes in `RuleVerifier`.

---

## 📄 License
MIT License.
