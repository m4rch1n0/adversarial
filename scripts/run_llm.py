"""LLM prompt injection experiments runner."""

import sys
import os
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aml_lab.llm.agent import ToolAgent, SYSTEM_PROMPTS
from aml_lab.llm.dataset import load_all_test_cases
from aml_lab.llm.eval import evaluate_test_case
from aml_lab.common.io import save_csv, save_metadata
from aml_lab.common.seed import set_seed


def prompt_user_choice(prompt, options):
    """Prompt user to select from options."""
    print(f"\n{prompt}")
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    while True:
        try:
            choice = int(input("Enter choice: ").strip())
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print(f"Invalid choice. Enter 1-{len(options)}")
        except KeyboardInterrupt:
            print("\nAborted by user")
            sys.exit(0)
        except ValueError:
            print("Invalid input")


def aggregate_attempts(results):
    """Aggregate results from multiple attempts."""
    total = len(results)
    emails_sent = sum(r["email_sent"] for r in results)
    attack_successes = sum(r["attack_success"] for r in results)
    false_actions = sum(r["false_action"] for r in results)
    asr = attack_successes / total if total > 0 else 0.0
    
    return {
        "attempts": total,
        "emails_sent": emails_sent,
        "attack_successes": attack_successes,
        "false_actions": false_actions,
        "asr": asr
    }


def save_results_incremental(output_dir, test_case, agg, system_prompt):
    """Append aggregated result for one test case to CSV."""
    csv_path = output_dir / "results.csv"
    row = [
        test_case.id,
        test_case.category,
        system_prompt,
        str(agg["attempts"]),
        str(agg["emails_sent"]),
        str(agg["attack_successes"]),
        f"{agg['asr']:.4f}"
    ]
    
    with open(csv_path, "a") as f:
        f.write(",".join(row) + "\n")


def compute_category_metrics(output_dir):
    """Compute aggregate metrics from results.csv."""
    csv_path = output_dir / "results.csv"
    
    with open(csv_path, "r") as f:
        lines = f.readlines()[1:]
    
    total_cases = len(lines)
    total_api_calls = 0
    total_asr_sum = 0.0
    
    for line in lines:
        parts = line.strip().split(",")
        if len(parts) >= 7:
            attempts = int(parts[3])
            asr = float(parts[6])
            total_api_calls += attempts
            total_asr_sum += asr
    
    overall_asr = total_asr_sum / total_cases if total_cases > 0 else 0.0
    
    return {
        "overall_asr": overall_asr,
        "total_test_cases": total_cases,
        "total_api_calls": total_api_calls
    }


def run_category(test_cases, agent, num_attempts, output_dir, system_prompt):
    """Run all test cases for a category."""
    print(f"\n{'='*60}")
    print(f"Processing {len(test_cases)} test cases...")
    print(f"{'='*60}")
    
    # Create CSV with header
    csv_path = output_dir / "results.csv"
    header = ["test_id", "category", "system_prompt", "attempts", "emails_sent", "attack_successes", "asr"]
    save_csv(csv_path, [], header)
    
    for i, tc in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] {tc.id}")
        
        attempt_results = []
        for attempt in range(1, num_attempts + 1):
            print(f"  Attempt {attempt}/{num_attempts}...", end=" ", flush=True)
            result = evaluate_test_case(agent, tc)
            attempt_results.append(result)
            status = "SUCCESS" if result["attack_success"] else ("PARTIAL (wrong recipient)" if result["email_sent"] else "FAILED")
            print(status)
        
        agg = aggregate_attempts(attempt_results)
        save_results_incremental(output_dir, tc, agg, system_prompt)
        print(f"  -> ASR: {agg['asr']:.2%} ({agg['attack_successes']}/{agg['attempts']})")


def main():
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not found in .env")
        sys.exit(1)
    
    print("="*60)
    print("LLM Prompt Injection Experiments")
    print("="*60)
    
    model = prompt_user_choice(
        "Select model:",
        ["gpt-4o-mini", "gpt-4.1-mini", "gpt-5-mini", "gpt-5"]
    )
    
    num_attempts = int(prompt_user_choice(
        "Number of attempts per test case:",
        ["1", "5", "10"]
    ))
    
    print("\nSystem prompt variants:")
    for i, (name, text) in enumerate(SYSTEM_PROMPTS.items(), 1):
        print(f"\n  {i}. {name.upper()}")
        print(f"     {text}")
    
    system_prompt = prompt_user_choice(
        "Select system prompt variant:",
        ["default", "restrictive", "strict"]
    )
    
    temp_display = "1.0 (fixed)" if model in ["gpt-5-mini", "gpt-5"] else "0.7"
    
    print(f"\nConfiguration:")
    print(f"  Model: {model}")
    print(f"  Attempts: {num_attempts}")
    print(f"  System prompt: {system_prompt}")
    print(f"  Temperature: {temp_display}")
    
    set_seed(123)
    
    print("\nLoading test cases from data/llm/...")
    all_cases = load_all_test_cases("data/llm")
    
    categories = {}
    for tc in all_cases:
        if tc.category not in categories:
            categories[tc.category] = []
        categories[tc.category].append(tc)
    
    print(f"Loaded {len(all_cases)} test cases across {len(categories)} categories:")
    for cat, cases in categories.items():
        print(f"  - {cat}: {len(cases)} cases")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nTimestamp: {timestamp}")
    
    agent = ToolAgent(model=model, api_key=api_key, temperature=0.7, system_prompt_variant=system_prompt)
    
    for category, test_cases in categories.items():
        print(f"\n{'#'*60}")
        print(f"# Category: {category.upper()}")
        print(f"{'#'*60}")
        
        output_dir = Path("results") / "llm" / model / category / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output: {output_dir}")
        
        run_category(test_cases, agent, num_attempts, output_dir, system_prompt)
        
        category_metrics = compute_category_metrics(output_dir)
        
        save_metadata(
            output_dir / "metadata.json",
            model=model,
            category=category,
            system_prompt=system_prompt,
            temperature=0.7,
            seed=123,
            num_attempts_per_case=num_attempts,
            test_cases_count=len(test_cases),
            category_metrics=category_metrics
        )
        
        print(f"\n Category '{category}' complete")
        print(f"  Overall ASR: {category_metrics['overall_asr']:.2%}")
        print(f"  Total API calls: {category_metrics['total_api_calls']}")
    
    print(f"\n{'='*60}")
    print("All experiments complete!")
    print(f"Results saved to: results/llm/{model}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

