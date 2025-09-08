from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Config:
    seed: int
    paths: dict
    train: dict
    compute: dict | None = None

def load_config(path: str | Path) -> Config:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    # Backward compatible: allow configs without `compute`
    if "compute" not in d:
        d["compute"] = {"backend": "auto"}
    return Config(**d)
