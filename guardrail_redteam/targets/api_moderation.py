"""
Black-box moderation API wrappers. These only expose a score/decision, not
gradients or internal logits -- used for the transfer study.

Both wrappers fail loudly (raise) if the relevant API key isn't set, rather
than silently no-oping, since a transfer study result is meaningless if the
target wasn't actually queried.
"""

import time
import requests

import config


class OpenAIModerationTarget:
    ENDPOINT = "https://api.openai.com/v1/moderations"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.OPENAI_API_KEY
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

    def score(self, text: str, retries: int = 3) -> float:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"input": text}
        for attempt in range(retries):
            resp = requests.post(self.ENDPOINT, headers=headers, json=payload, timeout=15)
            if resp.status_code == 200:
                result = resp.json()["results"][0]
                # max across category scores as a single scalar
                return max(result["category_scores"].values())
            time.sleep(2 ** attempt)
        resp.raise_for_status()

    def is_flagged(self, text: str) -> bool:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        resp = requests.post(self.ENDPOINT, headers=headers, json={"input": text}, timeout=15)
        resp.raise_for_status()
        return resp.json()["results"][0]["flagged"]


class PerspectiveAPITarget:
    ENDPOINT = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or config.PERSPECTIVE_API_KEY
        if not self.api_key:
            raise RuntimeError("PERSPECTIVE_API_KEY not set")

    def score(self, text: str, retries: int = 3) -> float:
        params = {"key": self.api_key}
        payload = {
            "comment": {"text": text},
            "requestedAttributes": {"TOXICITY": {}},
            "languages": ["en"],
        }
        for attempt in range(retries):
            resp = requests.post(self.ENDPOINT, params=params, json=payload, timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                return data["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            time.sleep(2 ** attempt)
        resp.raise_for_status()

    def is_flagged(self, text: str, threshold: float = 0.7) -> bool:
        return self.score(text) >= threshold


def get_enabled_api_targets():
    """Returns {name: target_instance} for every API target with a key set."""
    targets = {}
    if not config.USE_API_TARGETS:
        return targets
    if config.ENABLED_API_TARGETS.get("openai_moderation"):
        targets["openai_moderation"] = OpenAIModerationTarget()
    if config.ENABLED_API_TARGETS.get("perspective"):
        targets["perspective"] = PerspectiveAPITarget()
    return targets
