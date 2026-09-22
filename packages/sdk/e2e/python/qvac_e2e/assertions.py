"""Named assertions: the checks that are more than "contains this string".

These replace the JavaScript-function expectations in the catalog, which
cannot cross the wire. Each one is written once per client, against a name
that is part of the shared vocabulary — so two clients checking
`loadedModelInfoShape` are checking the same thing, not each their own idea
of it.

`args` carries the step's `with` block, already reference-resolved. That is
what lets a check compare the result against something the test set up rather
than only against a constant.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .result import StepResult


def length_is(value: Any, args: dict[str, Any]) -> StepResult:
    """The collection has exactly the expected number of elements.

    Deliberately generic: `topK: 1 must truncate to one result` is the same
    check as "this many blocks came back", and a registry of one-off names
    would defeat the point of a shared vocabulary.
    """
    if not isinstance(value, list):
        return StepResult.fail(f"expected an array, got {type(value).__name__}")
    expected = int(args.get("length", -1))
    if len(value) != expected:
        return StepResult.fail(f"expected {expected} element(s), got {len(value)}")
    return StepResult.ok(f"{len(value)} element(s)")


def loaded_model_info_shape(value: Any, args: dict[str, Any]) -> StepResult:
    """`getLoadedModelInfo` returned a record describing the model we loaded.

    Mirrors the checks the TypeScript executor made inline, so the migrated
    test asserts exactly as much as it did before.
    """
    if not isinstance(value, dict):
        return StepResult.fail(f"expected an object, got {type(value).__name__}")

    expected_model_id = args.get("expectedModelId")
    checks = {
        "modelIdMatches": value.get("modelId") == expected_model_id,
        "handlersIsList": isinstance(value.get("handlers"), list),
        "modelTypePresent": isinstance(value.get("modelType"), str)
        and len(value.get("modelType") or "") > 0,
    }

    expected_handler = args.get("handlerIncludes")
    if expected_handler is not None:
        handlers = value.get("handlers") or []
        checks["handlerPresent"] = expected_handler in handlers

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        return StepResult.fail(
            f"loadedModelInfoShape failed: {', '.join(failed)} "
            f"(modelId={value.get('modelId')}, modelType={value.get('modelType')}, "
            f"handlers={value.get('handlers')})"
        )

    return StepResult.ok(
        f"modelType={value.get('modelType')}, handlers={len(value.get('handlers') or [])}"
    )


ASSERTIONS: dict[str, Callable[[Any, dict[str, Any]], StepResult]] = {
    "lengthIs": length_is,
    "loadedModelInfoShape": loaded_model_info_shape,
}
