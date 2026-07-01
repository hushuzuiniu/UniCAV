# models/antiberta2_model.py
from __future__ import annotations

from typing import List, Optional, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import RoFormerForMaskedLM, RoFormerTokenizer

from .base_model import BaseModel


class Antiberta2Model(BaseModel):
    """Antibody-specific AntiBERTa2 embedding adapter for VH/VL sequences."""

    def __init__(
        self,
        model_name: str,
        model_path: str,
        device: Optional[str] = None,
        max_length: int = 512,
        chain_token: Optional[str] = None,
    ) -> None:
        super().__init__(model_name, model_path)
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.chain_token = chain_token
        self.tokenizer = RoFormerTokenizer.from_pretrained(model_path)
        self.model = RoFormerForMaskedLM.from_pretrained(model_path).to(self.device).eval()
        hidden_size = int(getattr(self.model.config, "hidden_size", 0) or 0)
        config_max_len = int(getattr(self.model.config, "max_position_embeddings", max_length) or max_length)
        self.model_max_len = min(int(max_length), config_max_len)
        print(
            f"[Antiberta2Model] loaded '{self.model_name}' from {self.model_path} "
            f"on {self.device}, hidden_size={hidden_size}, max_len={self.model_max_len}"
        )

    @property
    def embedding_dim(self) -> int:
        return int(getattr(self.model.config, "hidden_size", 0) or 0)

    @staticmethod
    def _clean_antibody_sequence(seq: str) -> str:
        valid = set("ACDEFGHIKLMNPQRSTVWYXBZUO")
        cleaned = "".join(ch if ch in valid else "X" for ch in str(seq).strip().upper().replace("*", ""))
        return cleaned or "X"

    @classmethod
    def _format_antibody_sequence(cls, seq: str, chain_token: Optional[str] = None) -> str:
        # AntiBERTa-style tokenizers expect amino-acid tokens separated by spaces.
        tokens = list(cls._clean_antibody_sequence(seq))
        if chain_token:
            tokens.insert(0, chain_token)
        return " ".join(tokens)

    def _pool_hidden(
        self,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        pool: str,
    ) -> torch.Tensor:
        mask = attention_mask.bool()
        special_mask = torch.zeros_like(mask)
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id
        if cls_id is not None:
            special_mask |= input_ids == cls_id
        if sep_id is not None:
            special_mask |= input_ids == sep_id
        if pad_id is not None:
            special_mask |= input_ids == pad_id
        token_mask = mask & ~special_mask

        if pool == "cls":
            return hidden[:, 0, :]
        if pool == "max":
            masked = hidden.masked_fill(~token_mask.unsqueeze(-1), torch.finfo(hidden.dtype).min)
            pooled = masked.max(dim=1).values
            empty = token_mask.sum(dim=1) == 0
            if empty.any():
                pooled[empty] = hidden[empty, 0, :]
            return pooled
        if pool != "mean":
            raise ValueError(f"Unknown pool type: {pool}")
        weights = token_mask.unsqueeze(-1).to(hidden.dtype)
        denom = weights.sum(dim=1).clamp(min=1.0)
        return (hidden * weights).sum(dim=1) / denom

    @torch.no_grad()
    def get_embedding(
        self,
        sequences: Union[str, List[str]],
        layer_name: str = None,
        batch_size: int = 16,
        pool: str = "mean",
        max_length: Optional[int] = None,
        return_numpy: bool = True,
        return_tokens: bool = False,
    ) -> Union[np.ndarray, torch.Tensor, List]:
        """Extract embeddings.

        Parameters
        ----------
        return_tokens : bool
            When True, return a list of per-sequence token embeddings with shape
            ``(L_aa, D)`` each, excluding CLS/SEP/PAD special tokens.  When
            False (default), return pooled ``(N, D)`` array.
        """
        del layer_name
        seq_list = [sequences] if isinstance(sequences, str) else list(sequences)
        config_max_len = int(getattr(self.model.config, "max_position_embeddings", self.model_max_len) or self.model_max_len)
        max_len = min(int(max_length or self.model_max_len), config_max_len)
        vocab_size = int(getattr(self.model.config, "vocab_size", len(self.tokenizer)) or len(self.tokenizer))
        outputs: List[torch.Tensor] = []

        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        for start in tqdm(range(0, len(seq_list), batch_size), desc="Getting AntiBERTa2 embedding", leave=False):
            batch = [
                self._format_antibody_sequence(seq, self.chain_token)
                for seq in seq_list[start:start + batch_size]
            ]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"]
            if input_ids.shape[1] > config_max_len:
                raise ValueError(
                    f"AntiBERTa2 tokenized length {input_ids.shape[1]} exceeds "
                    f"max_position_embeddings={config_max_len}."
                )
            min_id = int(input_ids.min().item())
            max_id = int(input_ids.max().item())
            if min_id < 0 or max_id >= vocab_size:
                raise ValueError(f"AntiBERTa2 token id out of range: min={min_id}, max={max_id}, vocab_size={vocab_size}")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            out = self.model(**encoded, output_hidden_states=True, return_dict=True)
            hidden = out.hidden_states[-1] if out.hidden_states is not None else out.logits

            if return_tokens:
                ids_dev = encoded["input_ids"]
                for i in range(hidden.shape[0]):
                    special = torch.zeros(ids_dev.shape[1], dtype=torch.bool, device=ids_dev.device)
                    if cls_id is not None:
                        special |= ids_dev[i] == cls_id
                    if sep_id is not None:
                        special |= ids_dev[i] == sep_id
                    if pad_id is not None:
                        special |= ids_dev[i] == pad_id
                    token_mask = (~special) & encoded["attention_mask"][i].bool()
                    seq_hidden = hidden.float()[i][token_mask]  # (L_aa, D)
                    outputs.append(seq_hidden.detach().cpu())
            else:
                pooled = self._pool_hidden(hidden.float(), encoded["attention_mask"], encoded["input_ids"], pool)
                outputs.append(pooled.detach().cpu())

        if return_tokens:
            if return_numpy:
                return [item.numpy() for item in outputs]
            return outputs
        result = torch.cat(outputs, dim=0) if outputs else torch.empty(0, self.embedding_dim)
        return result.numpy() if return_numpy else result
