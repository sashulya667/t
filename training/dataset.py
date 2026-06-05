"""Load JSONL training rows as prompt/completion pairs for SFT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

# Matches the suffix of input_text in training_data/*.jsonl
RESPONSE_TEMPLATE = "### C++:\n"


def split_prompt_completion(row: dict[str, Any]) -> tuple[str, str]:
    if "input_text" in row and "target_text" in row:
        return row["input_text"], row["target_text"]
    if "text" in row:
        text = row["text"]
        idx = text.rfind(RESPONSE_TEMPLATE)
        if idx == -1:
            raise ValueError("text field missing C++ response marker")
        return text[: idx + len(RESPONSE_TEMPLATE)], text[idx + len(RESPONSE_TEMPLATE) :]
    raise KeyError("Row must have input_text+target_text or text")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_dataset(path: Path) -> Dataset:
    rows = load_jsonl(path)
    prompts: list[str] = []
    completions: list[str] = []
    ids: list[str] = []
    for i, row in enumerate(rows):
        prompt, completion = split_prompt_completion(row)
        prompts.append(prompt)
        completions.append(completion)
        ids.append(row.get("id", str(i)))
    return Dataset.from_dict({"prompt": prompts, "completion": completions, "id": ids})
