"""
Validator Agent - The Self-Correction Loop.

Verifies each claim against its source document.
If a claim is not supported, it rewrites it or flags it.
This is what makes small models reliable for serious research.
"""

import json
from langchain_ollama import ChatOllama
from research_cli.config import Config
from research_cli.tools.scraper import scrape_page


class ValidatorAgent:
    """
    Fact-checks individual claims against source documents.

    For each claim from a researcher:
    1. Re-scrape the source URL to get fresh content.
    2. Ask the model: "Does this source support this claim? YES/NO."
    3. If NO, rewrite the claim to be accurate or remove it.
    """

    def __init__(self, config: Config | None = None):
        """Initialize the validator with a small model."""
        self.config = config or Config()
        self.config.initialize()
        self.model = ChatOllama(
            model=self.config.model.model_name,
            temperature=0.0,
            base_url=self.config.model.base_url,
        )
        self.validator_config = self.config.validator

    def validate_claim(self, claim: str, source_url: str) -> dict:
        """
        Validate a single claim against its source.

        Parameters:
            claim: The claim to verify.
            source_url: The URL of the source document.

        Returns:
            Dict with supported, reason, and corrected_claim.
        """
        if not source_url:
            return {
                "supported": False,
                "reason": "No source URL provided for verification.",
                "corrected_claim": "",
            }
        source_text = scrape_page(source_url, max_tokens=3000, config=self.config)
        if source_text.startswith("Error"):
            return {
                "supported": False,
                "reason": f"Could not access source: {source_text}",
                "corrected_claim": "",
            }
        prompt = self._build_prompt(claim, source_text)
        response = self.model.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        return self._parse_response(content)

    def validate_batch(self, findings: list[dict]) -> list[dict]:
        """
        Validate a batch of findings.

        Parameters:
            findings: List of finding dicts with claim and source_url.

        Returns:
            Updated findings with validated flag set.
        """
        validated = []
        for finding in findings:
            result = self.validate_claim(
                finding.get("claim", ""),
                finding.get("source_url", ""),
            )
            finding["validated"] = result.get("supported", False)
            if not result.get("supported", False) and result.get("corrected_claim"):
                finding["claim"] = result["corrected_claim"]
            finding["validation_reason"] = result.get("reason", "")
            validated.append(finding)
        return validated

    def _build_prompt(self, claim: str, source_text: str) -> str:
        """Build the validator prompt."""
        return f"""{self.validator_config.system_prompt}

Claim to verify: "{claim}"

Source document:
{source_text[:3000]}

Does the source document support this claim? Output ONLY valid JSON:"""

    def _parse_response(self, content: str) -> dict:
        """Parse the validator's JSON response."""
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        try:
            data = json.loads(content)
            return {
                "supported": data.get("supported", False),
                "reason": data.get("reason", ""),
                "corrected_claim": data.get("corrected_claim", ""),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            lower = content.lower()
            supported = "yes" in lower and "no" not in lower.split("yes")[0]
            return {
                "supported": supported,
                "reason": "Could not parse response, inferred from text.",
                "corrected_claim": "",
            }
