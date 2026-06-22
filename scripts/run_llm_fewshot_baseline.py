"""Baseline B6: GPT-style few-shot requirement classification.

This script evaluates a few-shot LLM baseline on the active PROMISE expanded
dataset protocol:

    data/exp/promise_exp.csv -> stratified 70/15/15 split

The default mode is retrieval few-shot: for each test sample, it selects the
most similar training examples per class using TF-IDF. This usually gives a
stronger LLM baseline than a single fixed prompt while still preventing data
leakage because examples are selected only from the training split.

Example:
    set OPENAI_API_KEY=sk-...
    python scripts/run_llm_fewshot_baseline.py --model gpt-4o-mini

Free local example with Ollama:
    ollama pull llama3.1:8b
    python scripts/run_llm_fewshot_baseline.py --model llama3.1:8b --base-url http://localhost:11434/v1

Paper-like fixed few-shot:
    python scripts/run_llm_fewshot_baseline.py --example-mode fixed --shots-per-class 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


PROMPT_VERSION = "b6_fewshot_v1"
DEFAULT_DATA_PATH = Path("data/exp/promise_exp.csv")
DEFAULT_OUTPUT_DIR = Path("results/metrics")
DEFAULT_CACHE_DIR = Path("results/cache/llm_baseline")
TEXT_COL = "RequirementText"
LABEL_COL = "class"


LABEL_DESCRIPTIONS = {
    "F": "Functional: behavior, feature, service, or action the system shall perform.",
    "FR": "Functional requirement.",
    "NFR": "Non-functional requirement.",
    "A": "Availability: uptime, accessibility, service continuity, or operational availability.",
    "FT": "Fault tolerance: ability to continue operating under faults, failures, or exceptions.",
    "L": "Legal: laws, regulations, standards, policy, license, or compliance constraints.",
    "LF": "Look and feel: visual appearance, interface style, display, layout, or aesthetics.",
    "MN": "Maintainability: ease of modification, maintenance, repair, diagnosis, or evolution.",
    "O": "Operational: operating environment, deployment, backup, administration, or procedures.",
    "PE": "Performance: speed, response time, throughput, capacity, resource usage, or timing.",
    "PO": "Portability: ability to run across platforms, operating systems, browsers, or devices.",
    "SC": "Scalability: ability to handle growth in users, load, data, or transactions.",
    "SE": "Security: confidentiality, authentication, authorization, encryption, privacy, or protection.",
    "US": "Usability: learnability, ease of use, user guidance, accessibility, or user experience.",
}

LABEL_ALIASES = {
    "FUNCTIONAL": "F",
    "FUNCTIONAL REQUIREMENT": "F",
    "NON-FUNCTIONAL": "NFR",
    "NON FUNCTIONAL": "NFR",
    "NON-FUNCTIONAL REQUIREMENT": "NFR",
    "NON FUNCTIONAL REQUIREMENT": "NFR",
    "AVAILABILITY": "A",
    "FAULT TOLERANCE": "FT",
    "FAULT-TOLERANCE": "FT",
    "LEGAL": "L",
    "LOOK AND FEEL": "LF",
    "LOOK & FEEL": "LF",
    "MAINTAINABILITY": "MN",
    "OPERATIONAL": "O",
    "PERFORMANCE": "PE",
    "PORTABILITY": "PO",
    "SCALABILITY": "SC",
    "SECURITY": "SE",
    "USABILITY": "US",
}


@dataclass(frozen=True)
class Example:
    text: str
    label: str
    similarity: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run GPT-style few-shot baseline B6 on PROMISE expanded."
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible endpoint, e.g. http://localhost:11434/v1 for Ollama.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key override. Local endpoints can use any dummy value.",
    )
    parser.add_argument("--task", choices=["full", "binary-fr-nfr", "top4-nfr"], default="full")
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shots-per-class", type=int, default=5)
    parser.add_argument("--example-mode", choices=["retrieval", "fixed"], default="retrieval")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=80)
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N test samples.")
    parser.add_argument("--dry-run", action="store_true", help="Print one prompt and exit without API calls.")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cache and call the model again.")
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--max-retries", type=int, default=4)
    return parser.parse_args()


def load_dataset(path: Path, task: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    missing = {TEXT_COL, LABEL_COL} - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df[[TEXT_COL, LABEL_COL]].dropna().copy()
    df[TEXT_COL] = df[TEXT_COL].astype(str).str.strip()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.strip().str.upper()
    df = df[df[TEXT_COL].ne("")]

    if task == "binary-fr-nfr":
        df[LABEL_COL] = np.where(df[LABEL_COL].eq("F"), "FR", "NFR")
    elif task == "top4-nfr":
        df = df[df[LABEL_COL].isin(["US", "SE", "O", "PE"])].copy()

    if df.empty:
        raise ValueError(f"No rows available after applying task={task!r}.")

    return df.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    seed: int,
    test_split: float,
    val_split: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if test_split <= 0 or val_split <= 0 or test_split + val_split >= 1:
        raise ValueError("test_split and val_split must be positive and sum to less than 1.")

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_split,
        random_state=seed,
        stratify=df[LABEL_COL],
    )
    adjusted_val = val_split / (1.0 - test_split)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val,
        random_state=seed,
        stratify=train_val_df[LABEL_COL],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def ordered_labels(labels: list[str]) -> list[str]:
    preferred = ["F", "FR", "NFR", "A", "FT", "L", "LF", "MN", "O", "PE", "PO", "SC", "SE", "US"]
    known = [label for label in preferred if label in labels]
    extra = sorted(label for label in labels if label not in known)
    return known + extra


def build_fixed_examples(
    train_df: pd.DataFrame,
    labels: list[str],
    shots_per_class: int,
    seed: int,
) -> list[Example]:
    examples: list[Example] = []
    for label in labels:
        class_df = train_df[train_df[LABEL_COL].eq(label)]
        sampled = class_df.sample(
            n=min(shots_per_class, len(class_df)),
            random_state=seed,
            replace=False,
        )
        examples.extend(
            Example(text=row[TEXT_COL], label=row[LABEL_COL])
            for _, row in sampled.iterrows()
        )
    random.Random(seed).shuffle(examples)
    return examples


class RetrievalExampleSelector:
    def __init__(self, train_df: pd.DataFrame, labels: list[str], shots_per_class: int):
        self.train_df = train_df.reset_index(drop=True)
        self.labels = labels
        self.shots_per_class = shots_per_class
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        self.train_matrix = self.vectorizer.fit_transform(self.train_df[TEXT_COL].tolist())

    def select(self, text: str) -> list[Example]:
        query = self.vectorizer.transform([text])
        similarities = cosine_similarity(query, self.train_matrix).ravel()
        examples: list[Example] = []

        for label in self.labels:
            label_indices = self.train_df.index[self.train_df[LABEL_COL].eq(label)].tolist()
            ranked = sorted(label_indices, key=lambda idx: similarities[idx], reverse=True)
            for idx in ranked[: self.shots_per_class]:
                row = self.train_df.iloc[idx]
                examples.append(
                    Example(
                        text=row[TEXT_COL],
                        label=row[LABEL_COL],
                        similarity=float(similarities[idx]),
                    )
                )

        examples.sort(key=lambda ex: (-1.0 if ex.similarity is None else -ex.similarity, ex.label))
        return examples


def label_block(labels: list[str]) -> str:
    lines = []
    for label in labels:
        description = LABEL_DESCRIPTIONS.get(label, "Requirement category.")
        lines.append(f"- {label}: {description}")
    return "\n".join(lines)


def format_examples(examples: list[Example]) -> str:
    formatted = []
    for idx, example in enumerate(examples, start=1):
        text = compact_text(example.text)
        formatted.append(f"Example {idx}\nRequirement: {text}\nLabel: {example.label}")
    return "\n\n".join(formatted)


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def build_messages(text: str, examples: list[Example], labels: list[str]) -> list[dict[str, str]]:
    system = (
        "You are an expert software requirements engineering classifier. "
        "Classify each requirement into exactly one allowed label. "
        "Use the label definitions and few-shot examples. "
        "Return JSON only, with this exact shape: {\"label\": \"<LABEL>\"}."
    )
    user = (
        "Allowed labels and definitions:\n"
        f"{label_block(labels)}\n\n"
        "Few-shot training examples:\n"
        f"{format_examples(examples)}\n\n"
        "Now classify this requirement.\n"
        f"Requirement: {compact_text(text)}\n\n"
        f"Allowed label values: {', '.join(labels)}\n"
        "Return only valid JSON."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def json_schema(labels: list[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "name": "requirement_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": labels,
                }
            },
            "required": ["label"],
            "additionalProperties": False,
        },
    }


def prediction_key(
    model: str,
    task: str,
    temperature: float,
    labels: list[str],
    text: str,
    examples: list[Example],
) -> str:
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "task": task,
        "temperature": temperature,
        "labels": labels,
        "text": compact_text(text),
        "examples": [{"text": compact_text(ex.text), "label": ex.label} for ex in examples],
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            cache[record["key"]] = record
    return cache


def append_cache(cache_path: Path, record: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_openai_client(base_url: str | None, api_key: str | None) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "The OpenAI SDK is not installed. Run: pip install openai"
        ) from exc

    resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
    if base_url and not resolved_api_key:
        resolved_api_key = "local"

    if not resolved_api_key:
        raise SystemExit("OPENAI_API_KEY is not set.")

    kwargs = {"api_key": resolved_api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def call_model(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    labels: list[str],
    temperature: float,
    max_output_tokens: int,
    max_retries: int,
) -> tuple[str, str]:
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            if hasattr(client, "responses"):
                try:
                    response = client.responses.create(
                        model=model,
                        input=messages,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        text={"format": json_schema(labels)},
                    )
                    raw_text = response.output_text
                except TypeError:
                    raw_text = call_chat_completions(
                        client,
                        model,
                        messages,
                        temperature,
                        max_output_tokens,
                    )
            else:
                raw_text = call_chat_completions(
                    client,
                    model,
                    messages,
                    temperature,
                    max_output_tokens,
                )
            return normalize_prediction(raw_text, labels), raw_text
        except Exception as exc:  # API errors vary by SDK version.
            last_error = exc
            if attempt >= max_retries:
                break
            time.sleep(min(30.0, 2.0**attempt))

    raise RuntimeError(f"OpenAI call failed after {max_retries + 1} attempts: {last_error}")


def call_chat_completions(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_output_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_output_tokens,
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content or ""


def normalize_prediction(raw_text: str, labels: list[str]) -> str:
    text = raw_text.strip()
    try:
        parsed = json.loads(text)
        candidate = str(parsed.get("label", "")).strip()
    except json.JSONDecodeError:
        match = re.search(r'"label"\s*:\s*"([^"]+)"', text)
        candidate = match.group(1).strip() if match else text

    candidate = candidate.strip().upper()
    candidate = re.sub(r"[^A-Z0-9 &-]", "", candidate)

    if candidate in labels:
        return candidate

    alias = LABEL_ALIASES.get(candidate)
    if alias in labels:
        return alias

    for label in labels:
        if re.search(rf"\b{re.escape(label)}\b", candidate):
            return label

    return "__INVALID__"


def evaluate(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "weighted_f1": weighted_f1,
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    df = load_dataset(args.data_path, args.task)
    train_df, val_df, test_df = split_dataset(df, args.seed, args.test_split, args.val_split)
    labels = ordered_labels(sorted(train_df[LABEL_COL].unique().tolist()))

    if args.limit is not None:
        test_df = test_df.head(args.limit).copy()

    if args.example_mode == "fixed":
        fixed_examples = build_fixed_examples(train_df, labels, args.shots_per_class, args.seed)
        selector = None
    else:
        fixed_examples = []
        selector = RetrievalExampleSelector(train_df, labels, args.shots_per_class)

    first_examples = fixed_examples if selector is None else selector.select(test_df.iloc[0][TEXT_COL])
    first_messages = build_messages(test_df.iloc[0][TEXT_COL], first_examples, labels)

    print("Baseline B6 LLM few-shot")
    print(f"Dataset: {args.data_path}")
    print(f"Task: {args.task}")
    print(f"Split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    print(f"Labels: {labels}")
    print(f"Model: {args.model}")
    print(f"Base URL: {args.base_url or 'OpenAI default'}")
    print(f"Few-shot mode: {args.example_mode}, shots/class={args.shots_per_class}")

    if args.dry_run:
        print("\n--- Dry-run prompt preview ---")
        print(json.dumps(first_messages, ensure_ascii=False, indent=2))
        return

    client = get_openai_client(args.base_url, args.api_key)
    run_name = (
        f"baseline_6_llm_{args.task}_{args.model}_{args.example_mode}_"
        f"{args.shots_per_class}shot_seed{args.seed}"
    )
    safe_run_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name)
    cache_path = args.cache_dir / f"{safe_run_name}.jsonl"
    cache = {} if args.no_cache else load_cache(cache_path)

    records: list[dict[str, Any]] = []
    for idx, row in test_df.iterrows():
        text = row[TEXT_COL]
        true_label = row[LABEL_COL]
        examples = fixed_examples if selector is None else selector.select(text)
        messages = build_messages(text, examples, labels)
        key = prediction_key(
            args.model,
            args.task,
            args.temperature,
            labels,
            text,
            examples,
        )

        if key in cache:
            pred_label = cache[key]["pred_label"]
            raw_output = cache[key]["raw_output"]
            from_cache = True
        else:
            pred_label, raw_output = call_model(
                client=client,
                model=args.model,
                messages=messages,
                labels=labels,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                max_retries=args.max_retries,
            )
            cache_record = {
                "key": key,
                "pred_label": pred_label,
                "raw_output": raw_output,
            }
            append_cache(cache_path, cache_record)
            cache[key] = cache_record
            from_cache = False
            time.sleep(args.sleep_seconds)

        records.append(
            {
                "row_index": int(idx),
                "text": text,
                "true_label": true_label,
                "pred_label": pred_label,
                "from_cache": from_cache,
            }
        )

        if (len(records) % 10 == 0) or (len(records) == len(test_df)):
            current_acc = accuracy_score(
                [record["true_label"] for record in records],
                [record["pred_label"] for record in records],
            )
            print(f"Progress: {len(records)}/{len(test_df)} | running accuracy={current_acc:.4f}")

    pred_df = pd.DataFrame(records)
    metrics = evaluate(pred_df["true_label"].tolist(), pred_df["pred_label"].tolist())
    metrics.update(
        {
            "model": args.model,
            "task": args.task,
            "prompt_version": PROMPT_VERSION,
            "example_mode": args.example_mode,
            "shots_per_class": args.shots_per_class,
            "seed": args.seed,
            "split": {
                "train": len(train_df),
                "val": len(val_df),
                "test": len(test_df),
            },
            "train_distribution": train_df[LABEL_COL].value_counts().sort_index().to_dict(),
            "val_distribution": val_df[LABEL_COL].value_counts().sort_index().to_dict(),
            "test_distribution": test_df[LABEL_COL].value_counts().sort_index().to_dict(),
            "labels": labels,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / f"{safe_run_name}_metrics.json"
    predictions_path = args.output_dir / f"{safe_run_name}_predictions.csv"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pred_df.to_csv(predictions_path, index=False)

    print("\nFinal metrics")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Predictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()
