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


def length_at_least(value: Any, args: dict[str, Any]) -> StepResult:
    """The collection has at least this many elements.

    The floor half of `length_is`: "the registry lists models" and "more than
    one result came back" are the same check with a different bound, and a test
    that pinned the exact count would fail whenever the registry grew.
    """
    if not isinstance(value, list):
        return StepResult.fail(f"expected an array, got {type(value).__name__}")
    minimum = int(args.get("length", 0))
    if len(value) < minimum:
        return StepResult.fail(f"expected at least {minimum}, got {len(value)}")
    return StepResult.ok(f"{len(value)} element(s)")


def numbers_in_range(value: Any, args: dict[str, Any]) -> StepResult:
    """Every element's named field sits inside the range.

    A probability is in [0,1] and a utilisation is in [0,1]; naming the bound
    in the test rather than the registry keeps one check answering both.
    """
    items = value if isinstance(value, list) else []
    field = str(args.get("field"))
    low, high = float(args.get("min", 0)), float(args.get("max", 1))
    for item in items:
        measured = item.get(field) if isinstance(item, dict) else None
        if not isinstance(measured, (int, float)) or not low <= measured <= high:
            return StepResult.fail(f"{field} is outside [{low}, {high}]: {measured!r}")
    return StepResult.ok(f"{len(items)} value(s) within [{low}, {high}]")


def sorted_descending_by(value: Any, args: dict[str, Any]) -> StepResult:
    """The list is ordered by the named field, largest first."""
    items = value if isinstance(value, list) else []
    field = str(args.get("field"))

    def reading(item: Any) -> float:
        return float((item or {}).get(field) or 0)

    for index in range(1, len(items)):
        if reading(items[index]) > reading(items[index - 1]):
            return StepResult.fail(f"not sorted by {field} at index {index}")
    return StepResult.ok(f"{len(items)} element(s) in order")


def sums_to(value: Any, args: dict[str, Any]) -> StepResult:
    """The named field sums to a value, within a tolerance.

    Softmax probabilities sum to one; the tolerance is what keeps that a claim
    about the model rather than about float accumulation order.
    """
    items = value if isinstance(value, list) else []
    field = str(args.get("field"))
    total = sum(float((item or {}).get(field) or 0) for item in items)
    expected = float(args.get("total", 1))
    tolerance = float(args.get("tolerance", 1e-3))
    if abs(total - expected) > tolerance:
        return StepResult.fail(
            f"{field} sums to {total}, not within {tolerance} of {expected}"
        )
    return StepResult.ok(f"{field} sums to {total}")


def system_resources_shape(value: Any, args: dict[str, Any]) -> StepResult:
    """A `getSystemResources` record is well formed.

    Three claims the executor made inline, kept together because they are one
    question about one record: every metric that reports `supported` says where
    the number came from and none that does not report a value anyway; a GPU is
    identified by an opaque id and never by the raw vendor/device identifiers,
    which is a privacy boundary rather than a shape detail; and a requested
    sample correlates with the capabilities it was taken against.

    `sample` says whether one was asked for -- a sample that arrives
    unrequested is as much a failure as one that is missing.
    """
    problems: list[str] = []
    record = value if isinstance(value, dict) else {}

    def metric(raw: Any, label: str) -> dict[str, Any]:
        measured = raw if isinstance(raw, dict) else {}
        if measured.get("status") == "supported":
            if not (measured.get("provenance") or {}).get("source"):
                problems.append(f"{label} has no provenance source")
        elif "value" in measured:
            problems.append(
                f"{label} exposes a value with status {measured.get('status')}"
            )
        return measured

    def utilization(raw: Any, label: str) -> None:
        measured = metric(raw, label)
        reading = measured.get("value")
        if measured.get("status") == "supported" and isinstance(reading, (int, float)):
            if reading < 0 or reading > 1:
                problems.append(f"{label} is outside 0..1: {reading}")

    capabilities = record.get("capabilities")
    if not capabilities:
        problems.append("capabilities are missing")
        capabilities = {}
    metric(capabilities.get("cpu"), "capabilities.cpu")
    metric(
        (capabilities.get("memory") or {}).get("totalBytes"),
        "capabilities.memory.totalBytes",
    )
    capability_gpus = metric(capabilities.get("gpus"), "capabilities.gpus")

    raw_identity = ("vendorId", "deviceId", "subsystemId", "revision")
    capability_ids: list[str] | None = None
    if capability_gpus.get("status") == "supported":
        gpus = capability_gpus.get("value") or []
        capability_ids = [str(gpu.get("id")) for gpu in gpus]
        for gpu in gpus:
            if not gpu.get("id"):
                problems.append("GPU has no opaque ID")
            for field in raw_identity:
                if field in gpu:
                    problems.append(f"GPU exposes private identity field {field}")

    if not args.get("sample"):
        if record.get("sample"):
            problems.append("sample returned when it was not requested")
        if problems:
            return StepResult.fail("; ".join(problems))
        return StepResult.ok("capabilities valid; sample omitted")

    sample = record.get("sample")
    if not sample:
        problems.append("requested sample is missing")
        return StepResult.fail("; ".join(problems))

    utilization(sample.get("cpu"), "sample.cpu")
    memory = sample.get("memory") or {}
    metric(memory.get("usedBytes"), "sample.memory.usedBytes")
    metric(memory.get("totalBytes"), "sample.memory.totalBytes")
    metric(memory.get("processUsedBytes"), "sample.memory.processUsedBytes")
    allowance = metric(
        memory.get("processAvailableBytes"), "sample.memory.processAvailableBytes"
    )
    if allowance.get("status") == "supported":
        reading = allowance.get("value")
        if isinstance(reading, (int, float)) and reading <= 0:
            problems.append(
                f"sample.memory.processAvailableBytes is not positive: {reading}"
            )
        scope = (allowance.get("provenance") or {}).get("scope")
        if scope != "process":
            problems.append(
                f"sample.memory.processAvailableBytes carries scope {scope}"
            )
    elif args.get("platform") == "ios":
        problems.append(
            f"sample.memory.processAvailableBytes is {allowance.get('status')} on iOS"
        )

    sample_gpus = metric(sample.get("gpus"), "sample.gpus")
    if capability_ids is not None and sample_gpus.get("status") == "supported":
        gpus = sample_gpus.get("value") or []
        if capability_ids != [str(gpu.get("id")) for gpu in gpus]:
            problems.append("capability and sample GPU IDs do not correlate")
        for gpu in gpus:
            label = f"sample.gpus.{gpu.get('id')}"
            utilization(gpu.get("compute"), f"{label}.compute")
            utilization(gpu.get("encode"), f"{label}.encode")
            utilization(gpu.get("decode"), f"{label}.decode")

    if problems:
        return StepResult.fail("; ".join(problems))
    return StepResult.ok("capabilities valid; sample valid")


def texts_by_id(value: Any, args: dict[str, Any]) -> StepResult:
    """Each named result's text carries what that result was asked for.

    A batch answers several prompts at once, so "the output contains both
    markers" is not the question -- either prompt could have produced both.
    `mode: "any"` is the looser form a vision prompt needs, where several words
    would each be a right answer.
    """
    results = value if isinstance(value, list) else []
    by_id = {
        r.get("id"): (r.get("final") or {}).get("contentText") or ""
        for r in results
        if isinstance(r, dict)
    }
    expected: dict[str, list[str]] = args.get("expect") or {}
    any_mode = args.get("mode") == "any"

    for prompt_id, terms in expected.items():
        text = by_id.get(prompt_id)
        if text is None:
            return StepResult.fail(f'no result for id "{prompt_id}": got {list(by_id)}')
        lower = text.lower()
        hits = [term for term in terms if term.lower() in lower]
        if (not hits) if any_mode else (len(hits) != len(terms)):
            missing = [term for term in terms if term not in hits]
            return StepResult.fail(
                f'"{prompt_id}" is missing '
                f"{'any of ' if any_mode else ''}{missing}: {text[:160]}"
            )
    return StepResult.ok(f"{len(expected)} id(s) matched")


def streamed_each_id(value: Any, args: dict[str, Any]) -> StepResult:
    """Every named result was streamed, not just delivered.

    The batch's final results look the same whether the text arrived in one
    frame or in fifty, so a streaming test that only read the finals would pass
    with streaming switched off.
    """
    events = value if isinstance(value, list) else []
    counts: dict[str, int] = {}
    for item in events:
        if not isinstance(item, dict):
            continue
        prompt_id = item.get("id")
        event = item.get("event") or {}
        if prompt_id is None:
            continue
        if event.get("type") == "contentDelta" and (event.get("text") or ""):
            counts[prompt_id] = counts.get(prompt_id, 0) + 1
    missing = [pid for pid in (args.get("ids") or []) if not counts.get(pid)]
    if missing:
        return StepResult.fail(f"no streamed content for: {', '.join(missing)}")
    return StepResult.ok(
        "streamed " + "/".join(str(c) for c in counts.values()) + " delta(s)"
    )


def no_tool_calls_for(value: Any, args: dict[str, Any]) -> StepResult:
    """These results made no tool call.

    The other half of `tool_call_shape`: a batch where one prompt declares a
    tool and another does not is only answered if the second one stayed quiet.
    """
    results = value if isinstance(value, list) else []
    ids = args.get("ids") or []
    for prompt_id in ids:
        calls = next(
            (
                (r.get("final") or {}).get("toolCalls") or []
                for r in results
                if isinstance(r, dict) and r.get("id") == prompt_id
            ),
            [],
        )
        if calls:
            names = ", ".join(str(c.get("name")) for c in calls)
            return StepResult.fail(
                f'"{prompt_id}" was expected to make no tool call, made: {names}'
            )
    return StepResult.ok(f"{len(ids)} id(s) stayed quiet")


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


def produced_audio(value: Any, args: dict[str, Any]) -> StepResult:
    """The run produced audio.

    `minSamples` is the bar: 1 for a normal synthesis, 0 for the tests that
    feed empty text and only care that the SDK handled it rather than crashing.
    The executors asserted a synthesised sentence -- "generated N samples" --
    against `type: string`, which every string satisfies, so they could not
    fail whatever the engine did. This asks the question they meant.
    """
    if isinstance(value, (list, tuple, bytes, bytearray)):
        samples = len(value)
    else:
        samples = 0
    floor = int(args.get("minSamples", 1))
    if samples < floor:
        return StepResult.fail(f"expected at least {floor} sample(s), got {samples}")
    return StepResult.ok(f"{samples} sample(s)")


def is_true(value: Any, args: dict[str, Any]) -> StepResult:
    """The value is exactly `True`.

    For the operations whose whole answer is "it worked": the executors turned
    that into the string "success" and matched it against `type: string`, which
    is satisfied by "failed" just as well.
    """
    if value is not True:
        return StepResult.fail(f"expected true, got {value!r}")
    return StepResult.ok("true")


def positive_integers(value: Any, args: dict[str, Any]) -> StepResult:
    """Every named field is a positive integer.

    Model hyper-parameters are the recurring case: a chunk size or an action
    dimension that arrives as 0, a float, or a string is a broken model
    description however well-formed the surrounding object looks.
    """
    record = value if isinstance(value, dict) else {}
    fields: list[str] = args.get("fields") or []
    for field in fields:
        measured = record.get(field)
        if not isinstance(measured, int) or isinstance(measured, bool) or measured <= 0:
            return StepResult.fail(
                f"{field} is not a positive integer (got {measured!r})"
            )
    return StepResult.ok(f"{len(fields)} field(s) positive")


def value_in(value: Any, args: dict[str, Any]) -> StepResult:
    """The value is one of a known set. `allowNull` admits "not reported"."""
    if value is None and args.get("allowNull"):
        return StepResult.ok("null")
    allowed = args.get("values") or []
    if value not in allowed:
        return StepResult.fail(f"{value!r} is not one of {allowed!r}")
    return StepResult.ok(str(value))


def field_equals(value: Any, args: dict[str, Any]) -> StepResult:
    """Two fields of the same object agree.

    For the invariants a result carries about itself -- a buffer whose length
    must equal the product of the dimensions reported beside it.
    """
    record = value if isinstance(value, dict) else {}
    left = record.get(str(args.get("field")))
    right = record.get(str(args.get("other")))
    if left != right:
        return StepResult.fail(
            f"{args.get('field')}={left!r} != {args.get('other')}={right!r}"
        )
    return StepResult.ok(f"{args.get('field')} == {args.get('other')}")


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
    "lengthAtLeast": length_at_least,
    "numbersInRange": numbers_in_range,
    "sortedDescendingBy": sorted_descending_by,
    "sumsTo": sums_to,
    "systemResourcesShape": system_resources_shape,
    "textsById": texts_by_id,
    "streamedEachId": streamed_each_id,
    "noToolCallsFor": no_tool_calls_for,
    "fieldsPresent": fields_present,
    "fieldsMatch": fields_match,
    "errorIsStructured": error_is_structured,
    "nonEmptyText": non_empty_text,
    "errorMatches": error_matches,
    "toolCallShape": tool_call_shape,
    "textBlockShape": text_block_shape,
    "timingStatsPresent": timing_stats_present,
    "producedAudio": produced_audio,
    "isTrue": is_true,
    "positiveIntegers": positive_integers,
    "valueIn": value_in,
    "fieldEquals": field_equals,
    "loadedModelInfoShape": loaded_model_info_shape,
}
