from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from vllm_omni.distributed.omni_connectors.kv_transfer_manager import KVCacheTransferData
from vllm_omni.distributed.omni_connectors.utils.kv_utils import get_kv_connector_key

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class SyntheticARKVConfig:
    num_layers: int
    seq_len: int
    num_heads: int
    head_dim: int
    dtype: str = "float16"
    device: str = "cpu"
    from_stage: str = "-1"
    to_stage: str = "0"
    from_tp: int = 1
    to_tp: int = 1
    request_id_prefix: str = "chatcmpl-"
    shm_threshold_bytes: int = 0

    def validate(self) -> None:
        for name in ("num_layers", "seq_len", "num_heads", "head_dim", "from_tp", "to_tp"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}.")
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {sorted(_DTYPES)}, got {self.dtype!r}.")
        if self.from_tp == self.to_tp:
            return
        larger = max(self.from_tp, self.to_tp)
        smaller = min(self.from_tp, self.to_tp)
        if larger % smaller != 0:
            raise ValueError(
                f"from_tp/to_tp must be evenly divisible for synthetic KV routing, "
                f"got from_tp={self.from_tp}, to_tp={self.to_tp}."
            )


@dataclass(frozen=True)
class SyntheticKVSendResult:
    request_id: str
    connector_keys: list[str]
    bytes_sent: int


def attach_synthetic_ar_kv_request_context(req: Any, ar_generated_text: str) -> None:
    req.extra_body.setdefault("ar_generated_text", ar_generated_text)


class SyntheticARKVProducer:
    """Writes synthetic AR KV payloads for DiT-only pressure tests."""

    def __init__(
        self,
        config: SyntheticARKVConfig,
        connector: Any | None = None,
    ) -> None:
        config.validate()
        self.config = config
        if connector is None:
            from vllm_omni.distributed.omni_connectors.connectors.shm_connector import SharedMemoryConnector

            connector = SharedMemoryConnector(
                {
                    "stage_id": config.from_stage,
                    "shm_threshold_bytes": config.shm_threshold_bytes,
                }
            )
        self._connector = connector
        self._layer_blocks = self._build_layer_blocks()

    def _build_layer_blocks(self) -> dict[str, list[torch.Tensor]]:
        dtype = _DTYPES[self.config.dtype]
        shape = (self.config.seq_len, self.config.num_heads, self.config.head_dim)
        return {
            "key_cache": [
                torch.empty(shape, dtype=dtype, device=self.config.device) for _ in range(self.config.num_layers)
            ],
            "value_cache": [
                torch.empty(shape, dtype=dtype, device=self.config.device) for _ in range(self.config.num_layers)
            ],
        }

    def server_request_id(self, benchmark_request_id: str) -> str:
        prefix = self.config.request_id_prefix
        if prefix and not benchmark_request_id.startswith(prefix):
            return f"{prefix}{benchmark_request_id}"
        return benchmark_request_id

    def connector_keys(self, request_id: str) -> list[str]:
        keys: list[str] = []
        for from_rank in range(self.config.from_tp):
            for to_rank in self._target_ranks(from_rank):
                keys.append(
                    get_kv_connector_key(
                        req_id=request_id,
                        from_stage=self.config.from_stage,
                        chunk_id=0,
                        from_rank=from_rank,
                        to_rank=to_rank,
                    )
                )
        return keys

    def _target_ranks(self, from_rank: int) -> list[int]:
        if self.config.from_tp == self.config.to_tp:
            return [from_rank]
        if self.config.from_tp > self.config.to_tp:
            return [from_rank // (self.config.from_tp // self.config.to_tp)]
        ratio = self.config.to_tp // self.config.from_tp
        base = from_rank * ratio
        return list(range(base, base + ratio))

    def prepare(self, benchmark_request_id: str) -> SyntheticKVSendResult:
        request_id = self.server_request_id(benchmark_request_id)
        keys = self.connector_keys(request_id)
        metadata = {
            "seq_len": self.config.seq_len,
            "synthetic_ar_kv": True,
            "synthetic_ar_kv_config": asdict(self.config),
        }
        payload = KVCacheTransferData(
            request_id=request_id,
            layer_blocks=self._layer_blocks,
            block_ids=[],
            metadata=metadata,
        )

        bytes_sent = 0
        for key in keys:
            success, size, _ = self._connector.put(
                from_stage=self.config.from_stage,
                to_stage=self.config.to_stage,
                put_key=key,
                data=payload,
            )
            if not success:
                raise RuntimeError(f"Failed to write synthetic AR KV payload for key {key}")
            bytes_sent += int(size)

        return SyntheticKVSendResult(
            request_id=request_id,
            connector_keys=keys,
            bytes_sent=bytes_sent,
        )
