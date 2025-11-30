"""
Dataset Preparation Script for Phi-4 Math Fine-tuning
Downloads, processes, and merges NuminaMath-TIR, NuminaMath-CoT, and AIME datasets.
Logs statistics to Weights & Biases.
"""

import json
import re
import random
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Any

import wandb
from datasets import load_dataset, Dataset
from tqdm import tqdm
from kaggle_secrets import UserSecretsClient

# Get W&B API key from Kaggle secrets
secrets = UserSecretsClient()
wandb_api_key = secrets.get_secret("WANDB_API_KEY")

# Login to W&B
import wandb
wandb.login(key=wandb_api_key)

# === Configuration ===
CONFIG = {
    "output_dir": "data",
    "sft_output": "sft_dataset.jsonl",
    "grpo_output": "grpo_dataset.jsonl",
    
    # Dataset sizes
    "numina_cot_sample": 40000,
    "aime_min_year": 2000,
    
    # Difficulty distribution (must sum to 1.0)
    "difficulty_split": {
        "easy": 0.20,
        "medium": 0.30,
        "hard": 0.50
    },
    
    # W&B
    "wandb_project": "phi4-math-data",
    "wandb_run_name": "dataset-preparation",
    
    # Random seed
    "seed": 42
}

# Prompt template for Phi-4
PROMPT_TEMPLATE = """<|user|>
{problem}
<|end|>
<|assistant|>
{solution}
<|end|>"""

GRPO_TEMPLATE = """<|user|>
{problem}
<|end|>
<|assistant|>
"""


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)


def extract_answer(solution: str) -> Optional[str]:
    """Extract boxed answer from solution."""
    patterns = [
        r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        r'\\fbox\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        r'[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+\$?([^\$\n,\.]+)',
        r'[Aa]nswer[:\s]+\$?([^\$\n,\.]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, solution)
        if matches:
            return matches[-1].strip()
    return None


def has_code_blocks(text: str) -> bool:
    """Check if text contains Python code blocks."""
    return bool(re.search(r'```python', text, re.IGNORECASE))


def classify_difficulty(example: Dict, source: str) -> str:
    """
    Classify problem difficulty based on source and solution length.
    
    Rules:
    - AIME, olympiads, IMO → hard
    - AMC, competition_math level 4-5 → medium/hard
    - Basic math, short solutions → easy
    """
    solution = example.get("solution", "")
    solution_len = len(solution)
    
    # Source-based classification
    source_lower = source.lower()
    problem_source = example.get("source", "").lower()
    
    # Hard sources
    hard_keywords = ["aime", "imo", "olympiad", "usamo", "putnam", "hmmt"]
    if any(kw in source_lower or kw in problem_source for kw in hard_keywords):
        return "hard"
    
    # Medium sources
    medium_keywords = ["amc", "mathcounts", "competition"]
    if any(kw in source_lower or kw in problem_source for kw in medium_keywords):
        return "medium" if solution_len < 2000 else "hard"
    
    # Length-based heuristic for others
    if solution_len < 500:
        return "easy"
    elif solution_len < 1500:
        return "medium"
    else:
        return "hard"


def format_example(problem: str, solution: str) -> str:
    """Apply prompt template to create formatted text."""
    return PROMPT_TEMPLATE.format(
        problem=problem.strip(),
        solution=solution.strip()
    )


def normalize_example(example: Dict, source: str) -> Dict[str, Any]:
    """Normalize example to unified schema."""
    # Handle different field names
    problem = example.get("problem") or example.get("question") or ""
    solution = example.get("solution") or example.get("answer") or ""
    
    # For TIR, solution might be in different field
    if not solution and "messages" in example:
        # Extract from chat format if present
        messages = example.get("messages", [])
        for msg in messages:
            if msg.get("role") == "assistant":
                solution = msg.get("content", "")
                break
    
    answer = extract_answer(solution)
    
    return {
        "text": format_example(problem, solution),
        "problem": problem.strip(),
        "solution": solution.strip(),
        "answer": answer,
        "source": source,
        "difficulty": classify_difficulty(example, source),
        "has_code": has_code_blocks(solution),
    }


def download_numina_tir() -> List[Dict]:
    """Download and process NuminaMath-TIR dataset."""
    print("\n📥 Downloading NuminaMath-TIR...")
    
    try:
        ds = load_dataset("AI-MO/NuminaMath-TIR", split="train")
    except Exception as e:
        print(f"   ⚠️ Error loading NuminaMath-TIR: {e}")
        print("   Trying alternative loading...")
        ds = load_dataset("AI-MO/NuminaMath-TIR", split="train", trust_remote_code=True)
    
    examples = []
    for item in tqdm(ds, desc="   Processing TIR"):
        normalized = normalize_example(item, "numina_tir")
        if normalized["problem"] and normalized["solution"]:
            examples.append(normalized)
    
    print(f"   ✅ Loaded {len(examples)} examples from NuminaMath-TIR")
    return examples


def download_numina_cot(sample_size: int) -> List[Dict]:
    """Download and process NuminaMath-CoT dataset with sampling."""
    print(f"\n📥 Downloading NuminaMath-CoT (sampling {sample_size})...")
    
    try:
        ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    except Exception as e:
        print(f"   ⚠️ Error loading NuminaMath-CoT: {e}")
        ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", trust_remote_code=True)
    
    # Process all first
    all_examples = []
    for item in tqdm(ds, desc="   Processing CoT"):
        normalized = normalize_example(item, "numina_cot")
        if normalized["problem"] and normalized["solution"]:
            all_examples.append(normalized)
    
    # Sample if needed
    if len(all_examples) > sample_size:
        examples = random.sample(all_examples, sample_size)
    else:
        examples = all_examples
    
    print(f"   ✅ Loaded {len(examples)} examples from NuminaMath-CoT (from {len(all_examples)} total)")
    return examples


def download_aime(min_year: int) -> tuple:
    """Download and process AIME dataset, filtering by year."""
    print(f"\n📥 Downloading AIME (year >= {min_year})...")
    
    try:
        ds = load_dataset("di-zhang-fdu/AIME_1983_2024", split="train")
    except Exception as e:
        print(f"   ❌ Could not load AIME dataset: {e}")
        return [], []
    
    print(f"   Columns: {ds.column_names}")
    
    examples = []
    grpo_examples = []
    
    for item in tqdm(ds, desc="   Processing AIME"):
        year = item.get("Year")
        
        # Filter by year
        if year is None or year < min_year:
            continue
        
        problem = item.get("Question", "")
        answer = str(item.get("Answer", ""))
        
        if not problem:
            continue
        
        # AIME dataset only has answers, no solutions
        # For SFT, we'd need solutions - skip SFT examples
        # For GRPO, we only need problem + answer ✓
        
        grpo_examples.append({
            "prompt": GRPO_TEMPLATE.format(problem=problem.strip()),
            "problem": problem.strip(),
            "answer": answer.strip(),
            "year": year,
            "source": "aime",
            "problem_id": item.get("ID", ""),
        })
    
    # AIME has no solutions, so no SFT examples from this dataset
    print(f"   ✅ Loaded {len(grpo_examples)} AIME GRPO examples (year >= {min_year})")
    print(f"   ⚠️ AIME has no solutions, only used for GRPO")
    
    return [], grpo_examples  # Empty SFT, only GRPO


def balance_by_difficulty(
    examples: List[Dict],
    target_split: Dict[str, float]
) -> List[Dict]:
    """
    Balance dataset to achieve target difficulty distribution.
    Uses stratified sampling.
    """
    print("\n⚖️ Balancing difficulty distribution...")
    
    # Group by difficulty
    by_difficulty = defaultdict(list)
    for ex in examples:
        by_difficulty[ex["difficulty"]].append(ex)
    
    # Current counts
    total = len(examples)
    current_dist = {k: len(v) / total for k, v in by_difficulty.items()}
    print(f"   Current distribution: { {k: f'{v:.1%}' for k, v in current_dist.items()} }")
    print(f"   Target distribution:  { {k: f'{v:.1%}' for k, v in target_split.items()} }")
    
    # Calculate target counts
    # Use the limiting factor (difficulty with least relative examples)
    ratios = {}
    for diff, target_pct in target_split.items():
        available = len(by_difficulty[diff])
        needed_for_full = total * target_pct
        ratios[diff] = available / needed_for_full if needed_for_full > 0 else float('inf')
    
    min_ratio = min(ratios.values())
    final_total = int(total * min_ratio)
    
    # Sample each difficulty
    balanced = []
    for diff, target_pct in target_split.items():
        target_count = int(final_total * target_pct)
        available = by_difficulty[diff]
        
        if len(available) >= target_count:
            sampled = random.sample(available, target_count)
        else:
            sampled = available  # Take all if not enough
            print(f"   ⚠️ Not enough {diff} examples: {len(available)} < {target_count}")
        
        balanced.extend(sampled)
    
    # Shuffle
    random.shuffle(balanced)
    
    # Verify final distribution
    final_dist = defaultdict(int)
    for ex in balanced:
        final_dist[ex["difficulty"]] += 1
    
    print(f"   Final distribution: { {k: f'{v/len(balanced):.1%} ({v})' for k, v in final_dist.items()} }")
    print(f"   Final dataset size: {len(balanced)}")
    
    return balanced


def compute_statistics(examples: List[Dict]) -> Dict[str, Any]:
    """Compute dataset statistics for logging."""
    stats = {
        "total_examples": len(examples),
        "by_source": defaultdict(int),
        "by_difficulty": defaultdict(int),
        "with_code": 0,
        "with_answer": 0,
        "solution_lengths": [],
        "problem_lengths": [],
    }
    
    for ex in examples:
        stats["by_source"][ex["source"]] += 1
        stats["by_difficulty"][ex["difficulty"]] += 1
        if ex.get("has_code"):
            stats["with_code"] += 1
        if ex.get("answer"):
            stats["with_answer"] += 1
        stats["solution_lengths"].append(len(ex.get("solution", "")))
        stats["problem_lengths"].append(len(ex.get("problem", "")))
    
    # Convert defaultdicts to regular dicts
    stats["by_source"] = dict(stats["by_source"])
    stats["by_difficulty"] = dict(stats["by_difficulty"])
    
    # Compute averages
    stats["avg_solution_length"] = sum(stats["solution_lengths"]) / len(stats["solution_lengths"]) if stats["solution_lengths"] else 0
    stats["avg_problem_length"] = sum(stats["problem_lengths"]) / len(stats["problem_lengths"]) if stats["problem_lengths"] else 0
    stats["pct_with_code"] = stats["with_code"] / len(examples) if examples else 0
    stats["pct_with_answer"] = stats["with_answer"] / len(examples) if examples else 0
    
    return stats


def log_to_wandb(sft_examples: List[Dict], grpo_examples: List[Dict], stats: Dict):
    """Log statistics and samples to Weights & Biases."""
    print("\n📊 Logging to Weights & Biases...")
    
    # Initialize W&B
    run = wandb.init(
        project=CONFIG["wandb_project"],
        name=CONFIG["wandb_run_name"],
        config=CONFIG
    )
    
    # Log summary metrics
    wandb.log({
        "sft_total_examples": stats["total_examples"],
        "grpo_total_examples": len(grpo_examples),
        "pct_with_code": stats["pct_with_code"],
        "pct_with_answer": stats["pct_with_answer"],
        "avg_solution_length": stats["avg_solution_length"],
        "avg_problem_length": stats["avg_problem_length"],
    })
    
    # Log distribution charts
    # By source
    source_data = [[k, v] for k, v in stats["by_source"].items()]
    wandb.log({
        "source_distribution": wandb.plot.bar(
            wandb.Table(data=source_data, columns=["source", "count"]),
            "source", "count", title="Examples by Source"
        )
    })
    
    # By difficulty
    diff_data = [[k, v] for k, v in stats["by_difficulty"].items()]
    wandb.log({
        "difficulty_distribution": wandb.plot.bar(
            wandb.Table(data=diff_data, columns=["difficulty", "count"]),
            "difficulty", "count", title="Examples by Difficulty"
        )
    })
    
    # Solution length histogram
    wandb.log({
        "solution_length_histogram": wandb.Histogram(stats["solution_lengths"])
    })
    
    # Log sample examples as table
    sample_size = min(100, len(sft_examples))
    sample_data = []
    for ex in random.sample(sft_examples, sample_size):
        sample_data.append([
            ex["source"],
            ex["difficulty"],
            ex["problem"][:500] + "..." if len(ex["problem"]) > 500 else ex["problem"],
            ex["solution"][:500] + "..." if len(ex["solution"]) > 500 else ex["solution"],
            ex["answer"] or "N/A",
            ex["has_code"]
        ])
    
    sample_table = wandb.Table(
        data=sample_data,
        columns=["source", "difficulty", "problem", "solution", "answer", "has_code"]
    )
    wandb.log({"sample_examples": sample_table})
    
    # Log datasets as artifacts
    sft_artifact = wandb.Artifact(
        name="sft_dataset",
        type="dataset",
        description=f"SFT dataset with {len(sft_examples)} examples"
    )
    sft_artifact.add_file(Path(CONFIG["output_dir"]) / CONFIG["sft_output"])
    run.log_artifact(sft_artifact)
    
    grpo_artifact = wandb.Artifact(
        name="grpo_dataset", 
        type="dataset",
        description=f"GRPO dataset with {len(grpo_examples)} AIME problems"
    )
    grpo_artifact.add_file(Path(CONFIG["output_dir"]) / CONFIG["grpo_output"])
    run.log_artifact(grpo_artifact)
    
    wandb.finish()
    print("   ✅ Logged to W&B")


def save_datasets(sft_examples: List[Dict], grpo_examples: List[Dict]):
    """Save datasets to JSONL files."""
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save SFT dataset
    sft_path = output_dir / CONFIG["sft_output"]
    print(f"\n💾 Saving SFT dataset to {sft_path}...")
    with open(sft_path, 'w', encoding='utf-8') as f:
        for ex in tqdm(sft_examples, desc="   Writing"):
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    # Save GRPO dataset
    grpo_path = output_dir / CONFIG["grpo_output"]
    print(f"💾 Saving GRPO dataset to {grpo_path}...")
    with open(grpo_path, 'w', encoding='utf-8') as f:
        for ex in grpo_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"   ✅ Saved {len(sft_examples)} SFT examples")
    print(f"   ✅ Saved {len(grpo_examples)} GRPO examples")


def main():
    """Main function to prepare all datasets."""
    print("=" * 60)
    print("🚀 Phi-4 Math Dataset Preparation")
    print("=" * 60)
    
    set_seed(CONFIG["seed"])
    
    # Download datasets
    tir_examples = download_numina_tir()
    cot_examples = download_numina_cot(CONFIG["numina_cot_sample"])
    aime_examples, grpo_examples = download_aime(CONFIG["aime_min_year"])
    
    # Merge all examples
    print("\n🔀 Merging datasets...")
    all_examples = tir_examples + cot_examples + aime_examples
    print(f"   Total before balancing: {len(all_examples)}")
    
    # Balance by difficulty
    balanced_examples = balance_by_difficulty(
        all_examples,
        CONFIG["difficulty_split"]
    )
    
    # Compute statistics
    stats = compute_statistics(balanced_examples)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 DATASET SUMMARY")
    print("=" * 60)
    print(f"   SFT Examples:  {stats['total_examples']:,}")
    print(f"   GRPO Examples: {len(grpo_examples):,}")
    print(f"   With Code:     {stats['pct_with_code']:.1%}")
    print(f"   With Answer:   {stats['pct_with_answer']:.1%}")
    print(f"   Avg Solution:  {stats['avg_solution_length']:.0f} chars")
    print("\n   By Source:")
    for source, count in stats["by_source"].items():
        print(f"      {source}: {count:,}")
    print("\n   By Difficulty:")
    for diff, count in stats["by_difficulty"].items():
        pct = count / stats["total_examples"] * 100
        print(f"      {diff}: {count:,} ({pct:.1f}%)")
    
    # Save datasets
    save_datasets(balanced_examples, grpo_examples)
    
    # Log to W&B
    try:
        log_to_wandb(balanced_examples, grpo_examples, stats)
    except Exception as e:
        print(f"\n⚠️ W&B logging failed: {e}")
        print("   Datasets saved locally. Run 'wandb login' to enable logging.")
    
    print("\n" + "=" * 60)
    print("✅ Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
