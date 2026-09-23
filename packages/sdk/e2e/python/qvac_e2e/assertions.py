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

import re

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


def fields_present(value: Any, args: dict[str, Any]) -> StepResult:
    """Every named field is present on the value.

    Replaces the inline "which required fields are missing" loops that several
    executors grew independently. Generic on purpose: the field list belongs to
    the test, not to the assertion registry.
    """
    if not isinstance(value, dict):
        return StepResult.fail(f"expected an object, got {type(value).__name__}")
    fields: list[str] = args.get("fields") or []
    # Key absence, not falsiness: JS checks `=== undefined`, so a field the SDK
    # reports as an explicit null counts as present there. Checking `is None`
    # here would fail the same record on Python and call it drift.
    missing = [field for field in fields if field not in value]
    if missing:
        return StepResult.fail(f"missing fields: {', '.join(missing)}")
    return StepResult.ok(f"{len(fields)} field(s) present")


def fields_match(value: Any, args: dict[str, Any]) -> StepResult:
    """Two records agree on the named fields.

    Compared as strings so a client that returns a number where another returns
    a numeric string is not reported as drift -- the question here is whether
    two views of the same record agree, not how each typed it.
    """
    left = value if isinstance(value, dict) else {}
    right = args.get("expected") or {}
    fields: list[str] = args.get("fields") or []
    mismatched = [
        field for field in fields if str(left.get(field)) != str(right.get(field))
    ]
    if mismatched:
        return StepResult.fail(
            "; ".join(
                f"{field}: {left.get(field)} != {right.get(field)}"
                for field in mismatched
            )
        )
    return StepResult.ok(f"{len(fields)} field(s) match")


def error_is_structured(value: Any, args: dict[str, Any]) -> StepResult:
    """The rejection carried machine-readable structure, not just a string.

    A chained cause or a present error code both answer that; which one a given
    SDK surfaces is an implementation choice, and pinning the test to one of
    them would make it a test of that choice rather than of the guarantee.
    """
    err = value if isinstance(value, dict) else {}
    code = err.get("code") or ""
    if not code and not err.get("hasCause"):
        return StepResult.fail(
            "rejection carried neither a code nor a cause: "
            f"{err.get('message') or '(no message)'}"
        )
    return StepResult.ok(
        f"hasCause={bool(err.get('hasCause'))}, code={code or '(none)'}"
    )


def non_empty_text(value: Any, args: dict[str, Any]) -> StepResult:
    """The value is a string with something in it.

    `expectedType: "string"` only asks about the type, and `minLength` in the
    expectation applies to arrays, so "it produced text" had no way to be said
    until now. Every generative category needs it.
    """
    if not isinstance(value, str):
        return StepResult.fail(f"expected a string, got {type(value).__name__}")
    if not value.strip():
        return StepResult.fail("expected text, got an empty string")
    return StepResult.ok(f"{len(value)} character(s)")


def error_matches(value: Any, args: dict[str, Any]) -> StepResult:
    """The rejection is the one the test meant, by code and by wording.

    `messageNotMatching` is the half that is easy to forget and the reason this
    is not just a `contains`: several error tests exist to prove a bad argument
    is rejected *by the SDK* rather than forwarded to the addon, and only the
    wording of the failure tells those two apart.
    """
    err = value if isinstance(value, dict) else {}
    message = err.get("message") or ""

    expected_code = args.get("code")
    if expected_code is not None and str(err.get("code")) != str(expected_code):
        return StepResult.fail(f"expected code {expected_code}, got {err.get('code')}")

    contains = args.get("messageContains")
    if contains is not None and str(contains).lower() not in message.lower():
        return StepResult.fail(f'message does not contain "{contains}": {message}')

    forbidden = args.get("messageNotMatching")
    if forbidden is not None and re.search(str(forbidden), message, re.IGNORECASE):
        return StepResult.fail(f"message matched the forbidden pattern: {message}")

    return StepResult.ok(f"code={err.get('code') or '(none)'}: {message[:120]}")


def tool_call_shape(value: Any, args: dict[str, Any]) -> StepResult:
    """The model made a structured tool call, and the right one.

    `declared` is the tools the test offered: a call naming something that was
    never declared is a failure however well-formed it looks, and that check is
    the reason this is not an ordinary field comparison.
    """
    calls = value if isinstance(value, list) else []
    if not calls:
        return StepResult.fail(
            "expected a structured tool call but the model made none"
        )

    declared = set(args.get("declared") or [])
    valid = [call for call in calls if call.get("name") in declared]
    if not valid:
        got = ", ".join(str(call.get("name") or "<unnamed>") for call in calls)
        return StepResult.fail(
            f"no tool call matched a declared tool. Got: [{got}], "
            f"declared: [{', '.join(sorted(declared))}]"
        )

    match = next((call for call in valid if call.get("name") == args.get("name")), None)
    if match is not None:
        call_args = match.get("arguments") or {}
        for key in args.get("argKeys") or []:
            if key not in call_args:
                return StepResult.fail(
                    f"tool call '{args.get('name')}' is missing argument "
                    f"'{key}': {call_args}"
                )

    return StepResult.ok(
        "tool call(s): " + ", ".join(str(call.get("name")) for call in valid)
    )


def text_block_shape(value: Any, args: dict[str, Any]) -> StepResult:
    """Every block carries the geometry a caller needs to place it.

    The OCR executors checked this inline; as a named assertion it is the same
    check on every client, which is the difference between two clients agreeing
    and two clients each having an opinion.
    """
    blocks = value if isinstance(value, list) else []
    for index, block in enumerate(blocks):
        block = block if isinstance(block, dict) else {}
        if not isinstance(block.get("text"), str):
            return StepResult.fail(f"block[{index}].text is not a string")
        bbox = block.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return StepResult.fail(f"block[{index}].bbox is not a 4-element array")
        for position, coordinate in enumerate(bbox):
            if not isinstance(coordinate, (int, float)) or isinstance(coordinate, bool):
                return StepResult.fail(
                    f"block[{index}].bbox[{position}] is not a number"
                )
        confidence = block.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            return StepResult.fail(f"block[{index}].confidence is not a number")
    return StepResult.ok(f"{len(blocks)} well-formed block(s)")


def timing_stats_present(value: Any, args: dict[str, Any]) -> StepResult:
    """The run reported how long it took.

    `field` names which timing to insist on, because the engines do not agree
    on what they measure -- and a test that only checks "stats exist" passes on
    an object full of nulls.
    """
    if not isinstance(value, dict):
        return StepResult.fail("stats is undefined, expected timing data")
    field = str(args.get("field") or "totalTime")
    measured = value.get(field)
    if not isinstance(measured, (int, float)) or isinstance(measured, bool):
        return StepResult.fail(f"expected stats.{field} > 0, got {measured!r}")
    if measured <= 0:
        return StepResult.fail(f"expected stats.{field} > 0, got {measured!r}")
    return StepResult.ok(f"{field}={measured}")


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
    "fieldsPresent": fields_present,
    "fieldsMatch": fields_match,
    "errorIsStructured": error_is_structured,
    "nonEmptyText": non_empty_text,
    "errorMatches": error_matches,
    "toolCallShape": tool_call_shape,
    "textBlockShape": text_block_shape,
    "timingStatsPresent": timing_stats_present,
    "loadedModelInfoShape": loaded_model_info_shape,
}
