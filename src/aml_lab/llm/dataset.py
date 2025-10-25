"""Test case management for LLM prompt injection experiments."""

import json
from pathlib import Path
from typing import List, Dict, Any


class TestCase:
    """Single test case for prompt injection evaluation."""
    
    def __init__(self, data: Dict[str, Any]):
        self.id = data["id"]
        self.category = data["category"]
        self.user_prompt = data["user_prompt"]
        self.description = data.get("description")
        self.context_length = data.get("context_length")
        self.system_prompt_variant = data.get("system_prompt_variant", "default")
        
        # ground truth
        gt = data["ground_truth"]
        self.should_send_email = gt["should_send_email"]
        self.expected_recipient = gt.get("expected_recipient")
        
        # attack payload (None for clean cases)
        payload = data.get("attack_payload")
        if payload:
            self.target_recipient = payload.get("target_recipient")
            self.target_subject = payload.get("target_subject")
        else:
            self.target_recipient = None
            self.target_subject = None
        
        # external document (for RAG/indirect injection)
        ext_doc = data.get("external_document")
        if ext_doc:
            self.external_doc_path = ext_doc.get("path")
            self.external_doc_type = ext_doc.get("type", "txt")
        else:
            self.external_doc_path = None
            self.external_doc_type = None
    
    def get_full_prompt(self) -> str:
        """Get prompt with external document content if present."""
        if not self.external_doc_path:
            return self.user_prompt
        
        # read external document
        if self.external_doc_type == "txt":
            with open(self.external_doc_path, "r") as f:
                doc_content = f.read()
        elif self.external_doc_type == "pdf":
            from pypdf import PdfReader
            reader = PdfReader(self.external_doc_path)
            doc_content = "\n".join(page.extract_text() for page in reader.pages)
        else:
            doc_content = f"[Unsupported document type: {self.external_doc_type}]"
        
        # inject document content into prompt
        return f"{self.user_prompt}\n\n---DOCUMENT---\n{doc_content}\n---END---"
    
    def __repr__(self) -> str:
        return f"TestCase(id={self.id}, category={self.category})"


def load_test_cases(path: str) -> List[TestCase]:
    """Load test cases from JSON file (expects a list)."""
    with open(path, "r") as f:
        data = json.load(f)
    
    return [TestCase(test_case) for test_case in data]


def load_all_test_cases(data_dir: str) -> List[TestCase]:
    """Load all test cases from category files in directory."""
    cases = []
    data_path = Path(data_dir)
    
    # load in specific order to keep categories organized
    for category_file in ["semantic_framing.json", "obfuscation.json", "jailbreak.json", "clean.json"]:
        file_path = data_path / category_file
        if file_path.exists():
            cases.extend(load_test_cases(str(file_path)))
    
    return cases

