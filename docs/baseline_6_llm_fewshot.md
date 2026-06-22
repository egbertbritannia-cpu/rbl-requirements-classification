# Baseline B6: GPT-Style Few-Shot LLM

This baseline evaluates a GPT-style few-shot classifier on the active dataset:

```text
data/exp/promise_exp.csv
```

The script uses the repository protocol: stratified 70/15/15 split from the
original `class` labels.

## Strong Default Run

```bash
set OPENAI_API_KEY=sk-...
python scripts/run_llm_fewshot_baseline.py --model gpt-4o-mini
```

Default behavior:

- task: full PROMISE expanded multi-class classification
- split: 70/15/15
- prompt: label definitions + few-shot examples
- example selection: retrieval few-shot from the training split
- shots per class: 5
- output parsing: JSON label only
- cache: `results/cache/llm_baseline/`

Results are saved to:

```text
results/metrics/
```

## Free Local Run

This does not reproduce the GPT-4o-mini baseline exactly, but it gives a free
open-weight LLM baseline using the same prompt, split, metrics, and evaluation
script.

With Ollama:

```bash
ollama pull llama3.1:8b
python scripts/run_llm_fewshot_baseline.py --model llama3.1:8b --base-url http://localhost:11434/v1
```

With LM Studio:

```bash
python scripts/run_llm_fewshot_baseline.py --model <loaded-model-id> --base-url http://localhost:1234/v1 --api-key local
```

For weaker hardware, reduce prompt size:

```bash
python scripts/run_llm_fewshot_baseline.py --model llama3.1:8b --base-url http://localhost:11434/v1 --shots-per-class 2
```

## Paper-Like Fixed Few-Shot

Use this when the goal is closer methodology reproduction instead of the
strongest prompt variant:

```bash
python scripts/run_llm_fewshot_baseline.py --example-mode fixed --shots-per-class 10
```

## Useful Options

Smoke test without API calls:

```bash
python scripts/run_llm_fewshot_baseline.py --dry-run
```

Evaluate only a small subset:

```bash
python scripts/run_llm_fewshot_baseline.py --limit 20
```

FR/NFR binary task:

```bash
python scripts/run_llm_fewshot_baseline.py --task binary-fr-nfr
```

Top-4 NFR task:

```bash
python scripts/run_llm_fewshot_baseline.py --task top4-nfr
```

Use a stronger or newer OpenAI model if available:

```bash
python scripts/run_llm_fewshot_baseline.py --model <model-name>
```

## Scientific Notes

- Few-shot examples are selected only from the training split.
- Validation and test labels are never used as prompt examples.
- The retrieval setting is stronger than a fixed prompt, so report it as
  "retrieval few-shot" rather than plain few-shot.
- For strict paper-style comparison, report the fixed few-shot setting.
