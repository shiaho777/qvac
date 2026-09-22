"""Resource key -> model, and the load/evict lifecycle around a test.

The table itself is no longer written here. `tests/catalog/resources.json` is
the shared copy every client reads, because a definition that says
`useModel: { deps: ["whisper"] }` only means the same thing on two clients if
both resolve `whisper` to the same model with the same config. What stays here
is the part that cannot be data: turning a `$const` name into this SDK's model
descriptor, and a `$asset` placeholder into a path on this platform.

An unknown key is an `incomplete` result, not a crash: the test applies, this
client cannot run it yet.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from tetherto.qvac_sdk import load_model, unload_model
from tetherto.qvac_sdk import models as model_constants

# The shared table, relative to this client. Overridable so a run can point at
# a catalog somewhere else without editing code.
_TABLE_PATH = Path(
    os.environ.get("QVAC_RESOURCE_TABLE")
    or Path(__file__).resolve().parents[2] / "tests" / "catalog" / "resources.json"
)

# Where `$asset` placeholders point on this platform.
_ASSET_ROOT = Path(
    os.environ.get("QVAC_ASSET_ROOT") or Path(__file__).resolve().parents[2] / "assets"
)

_PLATFORM = os.environ.get("QVAC_RESOURCE_PLATFORM", "desktop")


def _load_table() -> dict[str, dict[str, Any]]:
    if not _TABLE_PATH.exists():
        return {}
    table = json.loads(_TABLE_PATH.read_text())
    # `on` narrows a key to the platforms that define it; absent means all.
    return {
        dep: entry
        for dep, entry in table.items()
        if _PLATFORM in entry.get("on", [_PLATFORM])
    }


RESOURCES: dict[str, dict[str, Any]] = _load_table()


class MissingConstantError(LookupError):
    pass


def _resolve(value: Any) -> Any:
    """Replace `$const` and `$asset` placeholders, recursively.

    Recursive because the placeholders nest: a companion model sits inside
    `config`, sometimes inside an object inside `config`.
    """
    if isinstance(value, list):
        return [_resolve(item) for item in value]
    if isinstance(value, dict):
        name = value.get("$const")
        if isinstance(name, str):
            constant = getattr(model_constants, name, None)
            if constant is None:
                raise MissingConstantError(
                    f'model constant "{name}" is missing from the Python registry'
                )
            return constant
        asset = value.get("$asset")
        if isinstance(asset, dict):
            return str(_ASSET_ROOT / asset["kind"] / asset["file"])
        return {key: _resolve(item) for key, item in value.items()}
    return value


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

    def _definition(self, dep: str) -> dict[str, Any]:
        definition = RESOURCES.get(dep)
        if definition is None:
            raise UnknownResourceError(
                f'resource "{dep}" is not defined for platform "{_PLATFORM}" in '
                f"{_TABLE_PATH.name}"
            )
        return definition

    def source_of(self, dep: str) -> dict[str, Any]:
        """What `loadModel` would be called with for this key, without calling it.

        A test that drives the load path itself needs the source as data, and
        the source is the one thing a definition cannot write down: it is a
        per-client constant. The shared table knows it.
        """
        definition = self._definition(dep)
        try:
            source = _resolve(definition.get("constant"))
        except MissingConstantError as error:
            raise UnknownResourceError(str(error)) from error
        out: dict[str, Any] = {}
        if source is not None:
            out["modelSrc"] = source
        if definition.get("modelSrc") is not None:
            out["modelSrc"] = definition["modelSrc"]
        if definition.get("type"):
            out["modelType"] = definition["type"]
        return out

    async def ensure_loaded(self, dep: str) -> str:
        existing = self._loaded.get(dep)
        if existing:
            return existing

        definition = self._definition(dep)
        try:
            constant = _resolve(definition.get("constant"))
            config = _resolve(definition.get("config"))
        except MissingConstantError as error:
            raise UnknownResourceError(str(error)) from error

        self._log(f"loading {dep}")
        model_id = await load_model(
            self._transport,
            model_src=constant if constant is not None else definition.get("modelSrc"),
            model_type=definition.get("type"),
            model_config=config,
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
