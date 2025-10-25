"""single working file for incremental test case development."""

import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aml_lab.llm.agent import ToolAgent, send_email_smtp
from aml_lab.llm.dataset import load_test_cases
from aml_lab.llm.eval import evaluate_test_case

load_dotenv()

# always use test.json as working file
json_file = "data/llm/test.json"
send_emails = "--send-emails" in sys.argv

# model selection
model = "gpt-4o-mini"
if "--model" in sys.argv:
    model_idx = sys.argv.index("--model") + 1
    if model_idx < len(sys.argv):
        model = sys.argv[model_idx]

print(f"Usage: python test_single_file.py [--send-emails] [--model gpt-4o-mini|gpt-4.1-mini|gpt-5-mini|gpt-5]\n")

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in .env")
    sys.exit(1)

if send_emails:
    smtp_host = os.getenv("MAILTRAP_SMTP_HOST")
    smtp_port = int(os.getenv("MAILTRAP_SMTP_PORT"))
    smtp_user = os.getenv("MAILTRAP_USER")
    smtp_password = os.getenv("MAILTRAP_PASSWORD")
    
    if not all([smtp_host, smtp_user, smtp_password]):
        print("ERROR: SMTP credentials missing in .env")
        sys.exit(1)

print(f"Testing: {json_file}")
print(f"Model: {model}")
if send_emails:
    print("[SMTP MODE] Emails will be sent to Mailtrap")

# load test cases
try:
    cases = load_test_cases(json_file)
    if not cases:
        print("No test cases found in test.json")
        sys.exit(0)
except Exception as e:
    print(f"ERROR loading test.json: {e}")
    sys.exit(1)

# run test case (only first one in test.json)
tc = cases[0]

# create agent with appropriate system prompt variant
agent = ToolAgent(model=model, api_key=api_key, temperature=0.7, system_prompt_variant=tc.system_prompt_variant)
print("=" * 60)
print(f"ID: {tc.id}")
print(f"Category: {tc.category}")
print(f"System prompt variant: {tc.system_prompt_variant}")
if tc.description:
    print(f"Description: {tc.description}")
if tc.external_doc_path:
    print(f"External doc: {tc.external_doc_path}")

# show full prompt
full_prompt = tc.get_full_prompt()
print("PROMPT:")
print("-" * 60)
print(full_prompt)
print("-" * 60)

# evaluate
print("\nRunning...")
eval_result = evaluate_test_case(agent, tc)

# show result
print("\nRESULT:")
print(f"  Email sent: {eval_result['email_sent']}")

if eval_result['email_sent']:
    print(f"  Recipient: {eval_result['recipient']}")
    print(f"  Subject: {eval_result['subject']}")
    
    # actually send email if flag is set
    if send_emails:
        print("\n  Sending via SMTP...")
        body = eval_result.get('body')
        success = send_email_smtp(
            recipient=eval_result['recipient'],
            subject=eval_result['subject'],
            body=body,
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password
        )
        if success:
            print("  Sent successfully to Mailtrap (waiting 9 seconds to avoid sandbox limit)")
            print("  Check: https://mailtrap.io/inboxes")
            time.sleep(9)
        else:
            print("  Send failed")

print(f"  Attack success: {eval_result['attack_success']}")
print(f"  False action: {eval_result['false_action']}")

if eval_result['response_content']:
    print(f"\n  Response preview:")
    preview = eval_result['response_content']
    print(f"  {preview}")

print("=" * 60)

