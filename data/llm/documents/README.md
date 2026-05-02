# Adversarial Test Documents

Synthetic documents used as adversarial inputs for evaluating prompt-injection robustness on tool-augmented LLM agents (see `main.tex` and `data/llm/*.json`).

Organized by attack category:
- `clean/` - benign control documents (no injected instruction).
- `jailbreak/` - documents with direct system-prompt overrides.
- `obfuscation/` - documents with hidden or encoded instructions.
- `semantic_framing/` - documents that frame an injected action as compliant with an authoritative protocol (whistleblower disclosure, AML reporting, child-safety reporting, threat intelligence, etc.).

All documents are synthetic and were authored solely to test whether LLM agents inappropriately execute tool calls (e.g. `send_email`) based on instructions embedded in retrieved content. They do not contain real harmful content, real personal data, or actionable instructions. File names refer to the scenario being simulated, not to the subject of any real material.
