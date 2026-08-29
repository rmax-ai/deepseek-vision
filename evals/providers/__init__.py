"""Provider registry for the evaluation harness.

Adapter modules are imported lazily on first use via ``get_provider`` so the
harness never pays the import cost of a provider it does not run.
"""

from __future__ import annotations

import importlib
from typing import Any

from ..protocol import MultimodalProvider

# name -> (import path, class name)
_PROVIDER_ENTRIES: dict[str, tuple[str, str]] = {
    "deepseek": ("evals.providers.deepseek", "DeepSeekProvider"),
}

PROVIDERS: dict[str, type[MultimodalProvider]] = {}


def get_provider(name: str, **kwargs: Any) -> MultimodalProvider:
    """Return an instance of the named provider.

    Raises ``KeyError`` listing the available provider names when ``name`` is
    unknown. The provider module is imported lazily on first use.
    """
    provider_type = PROVIDERS.get(name)
    if provider_type is None:
        entry = _PROVIDER_ENTRIES.get(name)
        if entry is None:
            available = ", ".join(sorted(_PROVIDER_ENTRIES))
            raise KeyError(
                f"unknown provider {name!r}; available: {available}"
            )
        module_path, class_name = entry
        module = importlib.import_module(module_path)
        provider_type = getattr(module, class_name)
        PROVIDERS[name] = provider_type
    return provider_type(**kwargs)
