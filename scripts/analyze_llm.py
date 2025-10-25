"""Analyze LLM prompt injection experiment results."""

import json
from pathlib import Path
from aml_lab.common.io import save_csv

RESULTS_DIR = Path("results/llm")
OUTPUT_DIR = RESULTS_DIR / "analysis"


def load_results():
    """Load all results from experiment runs."""
    runs = []
    
    for model_dir in RESULTS_DIR.iterdir():
        if model_dir.name == "analysis":
            continue
        
        for category_dir in model_dir.iterdir():
            for run_dir in category_dir.iterdir():
                with open(run_dir / "metadata.json") as f:
                    metadata = json.load(f)
                
                runs.append({
                    "model": model_dir.name,
                    "category": category_dir.name,
                    "system_prompt": metadata["system_prompt"],
                    "overall_asr": metadata["category_metrics"]["overall_asr"]
                })
    
    return runs


def build_aggregated_data(runs, group_by):
    """Aggregate ASR data by model and grouping key."""
    data = {}
    for run in runs:
        if run["category"] == "clean":
            continue
        
        model = run["model"]
        key = run[group_by]
        
        if model not in data:
            data[model] = {}
        if key not in data[model]:
            data[model][key] = []
        
        data[model][key].append(run["overall_asr"])
    
    return data


def build_table_model_vs_category(runs):
    """Build table: Model x Category ASR."""
    data = build_aggregated_data(runs, "category")
    categories = ["semantic_framing", "obfuscation", "jailbreak"]
    
    rows = []
    for model in sorted(data.keys()):
        row = [model]
        category_asrs = []
        
        for cat in categories:
            avg_asr = sum(data[model][cat]) / len(data[model][cat])
            row.append(f"{avg_asr:.4f}")
            category_asrs.append(avg_asr)
        
        overall_avg = sum(category_asrs) / len(category_asrs)
        row.append(f"{overall_avg:.4f}")
        rows.append(row)
    
    header = ["model", "semantic_framing", "obfuscation", "jailbreak", "avg_asr"]
    return header, rows


def build_table_model_vs_prompt(runs):
    """Build table: Model x System Prompt ASR."""
    data = build_aggregated_data(runs, "system_prompt")
    prompts = ["default", "restrictive", "strict"]
    
    rows = []
    for model in sorted(data.keys()):
        row = [model]
        for prompt in prompts:
            avg_asr = sum(data[model][prompt]) / len(data[model][prompt])
            row.append(f"{avg_asr:.4f}")
        rows.append(row)
    
    return ["model", "default", "restrictive", "strict"], rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {RESULTS_DIR}...")
    runs = load_results()
    print(f"Loaded {len(runs)} experiment runs\n")
    
    header_a, rows_a = build_table_model_vs_category(runs)
    save_csv(OUTPUT_DIR / "table_model_vs_category.csv", rows_a, header_a)
    
    header_b, rows_b = build_table_model_vs_prompt(runs)
    save_csv(OUTPUT_DIR / "table_model_vs_prompt.csv", rows_b, header_b)
    
    print(f"Results saved in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

