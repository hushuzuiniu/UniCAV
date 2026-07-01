from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence


@dataclass
class SequenceEmbeddingOptions:
    batch_size: int = 16
    pool: str = "mean"
    layer_name: str | None = None
    max_length: int | None = None
    hidden_size: int | None = None
    is_cnn: bool = False


_MODEL_LOADERS: dict[str, Callable[..., tuple[Any, SequenceEmbeddingOptions]]] = {}


def register_model(name: str, loader: Callable[..., tuple[Any, SequenceEmbeddingOptions]]) -> None:
    _MODEL_LOADERS[str(name)] = loader


def list_registered_models() -> list[str]:
    return sorted(_MODEL_LOADERS)


def load_embedder(
    model_name: str,
    model_dir: str | None = None,
    *,
    check_conda_env: bool = False,
    strict_conda_env: bool = False,
    **kwargs: Any,
) -> tuple[Any, SequenceEmbeddingOptions]:
    del check_conda_env, strict_conda_env
    try:
        loader = _MODEL_LOADERS[model_name]
    except KeyError as exc:
        raise KeyError(f"Model {model_name!r} is not registered in this UniCAV Space.") from exc
    return loader(model_name=model_name, model_dir=model_dir, **kwargs)


def extract_embeddings(
    model: Any,
    sequences: Sequence[str],
    *,
    options: SequenceEmbeddingOptions,
    **overrides: Any,
) -> Any:
    kwargs: dict[str, Any] = {
        "batch_size": overrides.pop("batch_size", options.batch_size),
        "pool": overrides.pop("pool", options.pool),
        "max_length": overrides.pop("max_length", options.max_length),
        "return_numpy": overrides.pop("return_numpy", True),
    }
    if options.layer_name is not None:
        kwargs["layer_name"] = options.layer_name
    kwargs.update(overrides)
    return model.get_embedding(list(sequences), **kwargs)
