"""Config loading.

One YAML file is the single source of truth for a run (model, loss,
optimizer, data, augmentation, precision, seed). Training and inference
must read the SAME resolved config so values never silently diverge
between the two (Architecture.md §11).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict that also supports attribute access (cfg.model.name) for
    convenience, while remaining a plain, JSON/YAML-serializable dict."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[item] = value
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    """Load a YAML config, optionally applying a flat dotted-key override
    dict such as {"optimizer.lr": 1e-4} from the CLI."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config(copy.deepcopy(raw))
    cfg["_config_path"] = str(path)
    if overrides:
        for dotted_key, value in overrides.items():
            _set_dotted(cfg, dotted_key, value)
    return cfg


def _set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            node[k] = {}
        node = node[k]
    node[keys[-1]] = value


def save_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(path, "w") as f:
        yaml.safe_dump(clean, f, sort_keys=False)
