"""Resource key -> model, and the load/evict lifecycle around a test.

This mirrors `tests/shared/resource-manager.ts`. It is duplicated per client
on purpose *for now*: the JS table lives inside the consumer entry files,
which an interpreter in another language cannot read. Moving the table into
the catalog as shared data is its own phase; until then keeping a small,
explicit copy here is honest about the duplication and cheap to delete.

Only the keys a migrated category needs are defined. An unknown key is an
`incomplete` result, not a crash: the test applies, this client cannot run it
yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tetherto.qvac_sdk import load_model, unload_model
from tetherto.qvac_sdk import models as model_constants


@dataclass(frozen=True)
class ResourceDefinition:
    """One entry of the shared resource table."""

    constant: str
    model_type: str
    config: dict[str, Any] | None = None


# Mirrors resources.define(...) in tests/desktop/consumer.ts. Grow this as
# categories migrate; it disappears when the table moves into the catalog.
RESOURCES: dict[str, ResourceDefinition] = {
    "embeddings": ResourceDefinition(
        constant="GTE_LARGE_FP16",
        model_type="llamacpp-embedding",
    ),
    "llm": ResourceDefinition(
        constant="LLAMA_3_2_1B_INST_Q4_0",
        model_type="llamacpp-completion",
        config={"verbosity": 0, "ctx_size": 2048},
    ),
}


class UnknownResourceError(LookupError):
    pass


class ResourceManager:
    """Loads models on demand and evicts what a test did not declare.

    The interpreter owns this lifecycle because the executors own it today: a
    test declares what it needs, everything else is evicted first, so a run is
    not sensitive to the order tests happen to arrive in.
    """

    def __init__(self, transport: Any, log) -> None:
        self._transport = transport
        self._log = log
        self._loaded: dict[str, str] = {}

    @property
    def transport(self) -> Any:
        return self._transport

    def _definition(self, dep: str) -> ResourceDefinition:
        definition = RESOURCES.get(dep)
        if definition is None:
            raise UnknownResourceError(
                f'resource "{dep}" is not in the Python resource table yet'
            )
        return definition

    async def ensure_loaded(self, dep: str) -> str:
        existing = self._loaded.get(dep)
        if existing:
            return existing

        definition = self._definition(dep)
        constant = getattr(model_constants, definition.constant, None)
        if constant is None:
            raise UnknownResourceError(
                f'model constant "{definition.constant}" is missing from the Python registry'
            )

        self._log(f"loading {dep} ({definition.constant})")
        model_id = await load_model(
            self._transport,
            model_src=constant,
            model_type=definition.model_type,
            model_config=definition.config,
        )
        self._loaded[dep] = model_id
        return model_id

    async def evict_all_except(self, keep: set[str]) -> None:
        for dep in [d for d in self._loaded if d not in keep]:
            model_id = self._loaded.pop(dep)
            self._log(f"evicting {dep}")
            try:
                await unload_model(self._transport, model_id)
            except Exception as error:  # noqa: BLE001 - eviction must not fail a test
                self._log(f"evicting {dep} failed (continuing): {error}")

    async def close(self) -> None:
        await self.evict_all_except(set())
