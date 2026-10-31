"""Run settings. Defaults are the values used for the reported results; environment
variables override them (e.g. PSIDUCK_QUANT=nf4 for smaller GPUs).
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Optional


def _opt_float(v: Optional[str]) -> Optional[float]:
    return float(v) if v not in (None, "", "none", "None") else None


@dataclass(frozen=True)
class Settings:
    model_name: str = "Qwen/Qwen3-8B"
    quantization: str = "bf16"  # "bf16", or "nf4" (bitsandbytes 4-bit) for smaller GPUs
    max_new_tokens: int = 2048  # same cap for the baseline and the latent decoder
    latent_steps: int = 30  # per delegate
    early_exit_cos: Optional[float] = None  # None = always run all latent steps
    thinking: bool = True  # Qwen3 thinking mode (its chat-template default, as in LatentMAS)
    temperature: float = 0.6  # 0 = greedy
    top_p: float = 0.95
    seed: int = 42
    gsm8k_budget_hours: float = 12.0  # time budget used to pick N from smoke timings

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            model_name=os.environ.get("PSIDUCK_MODEL", d.model_name),
            quantization=os.environ.get("PSIDUCK_QUANT", d.quantization),
            max_new_tokens=int(os.environ.get("PSIDUCK_MAX_NEW_TOKENS", d.max_new_tokens)),
            latent_steps=int(os.environ.get("PSIDUCK_LATENT_STEPS", d.latent_steps)),
            early_exit_cos=_opt_float(os.environ.get("PSIDUCK_EARLY_EXIT_COS")),
            thinking=os.environ.get("PSIDUCK_THINKING", str(d.thinking)).lower() in ("1", "true", "yes"),
            temperature=float(os.environ.get("PSIDUCK_TEMPERATURE", d.temperature)),
            top_p=float(os.environ.get("PSIDUCK_TOP_P", d.top_p)),
            seed=int(os.environ.get("PSIDUCK_SEED", d.seed)),
            gsm8k_budget_hours=float(os.environ.get("PSIDUCK_GSM8K_BUDGET_HOURS", d.gsm8k_budget_hours)),
        )

    def as_dict(self) -> dict:
        return asdict(self)
