from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass
class Config:
    seed: int
    paths: dict
    train: dict

def load_config(path: str | Path) -> Config:
    with open(path, "r") as f:
        d = yaml.safe_load(f)
    return Config(**d)
