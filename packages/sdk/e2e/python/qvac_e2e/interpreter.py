"""Executes a declarative test body.

The interpreter is the whole cost of a new client: it grows with the size of
the step vocabulary, not with the size of the catalog. An operation it does
not implement yet produces `incomplete` -- the test applies to this client,
the client simply cannot run it -- which keeps a thin client from looking
green.

Calls go through the generated typed methods on purpose, never straight to
the transport. The point of the exercise is to prove the *client* behaves like
the JS client; bypassing it would prove only that the worker works.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from tetherto.qvac_sdk import (
    BciTranscribeRequest,
    ClassifyRequest,
    DownloadAssetRequest,
    EmbedRequest,
    FinetuneRequest,
    GetLoadedModelInfoRequest,
    GetModelInfoRequest,
    GetSystemResourcesRequest,
    HeartbeatRequest,
    RagRequest,
    ResumeRequest,
    StateRequest,
    SuspendRequest,
    TranscribeRequest,
    VectorIndexRequest,
    bci_transcribe,
    cancel,
    classify,
    completion,
    delete_cache,
    download_asset,
    embed,
    finetune,
    get_loaded_model_info,
    get_model_info,
    get_system_resources,
    heartbeat,
    invoke_plugin,
    load_model,
    model_registry_get_model,
    model_registry_list,
    model_registry_search,
    rag,
    resume,
    state,
    suspend,
    transcribe,
    translate,
    unload_model,
    vector_index,
    vla,
    vla_hparams,
    vla_set_embodiment,
)

from .assertions import ASSERTIONS
from .resources import ASSET_ROOT, ResourceManager, UnknownResourceError
from .result import StepResult
from .validation import validate


# method name (as it appears in the contract manifest) -> how to call it here.
#
# Each entry takes the transport and the step's already-resolved params and
# adapts them to whatever this client's signature happens to be: a request
# model for the generated stubs, keyword arguments for the hand-written
# ergonomic wrappers, positional arguments for `model_registry_get_model`.
# Adapting is the binding's whole job -- the contract name and the params
# object are what the two clients agree on, not the calling convention.
#
# Deliberately explicit rather than reflective: a typo in a step should be an
# `incomplete` with a clear reason, not an attribute error deep in a stream.
def _request(model: Any, method: str, call: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a generated stub that takes a validated request model."""

    def invoke(transport: Any, params: dict[str, Any]) -> Any:
        return call(transport, model.model_validate({"type": method, **params}))

    return invoke


CALLS: dict[str, Callable[[Any, dict[str, Any]], Any]] = {
    # --- inference, request/reply -------------------------------------------
    "embed": _request(EmbedRequest, "embed", embed),
    "classify": _request(ClassifyRequest, "classify", classify),
    "transcribe": _request(TranscribeRequest, "transcribe", transcribe),
    "bciTranscribe": _request(BciTranscribeRequest, "bciTranscribe", bci_transcribe),
    "vla": lambda transport, params: vla(transport, **_snake(params)),
    "vlaHparams": lambda transport, params: vla_hparams(
        transport, model_id=params["modelId"]
    ),
    "vlaSetEmbodiment": lambda transport, params: vla_set_embodiment(
        transport, model_id=params["modelId"], embodiment=params["embodiment"]
    ),
    # --- models --------------------------------------------------------------
    "loadModel": lambda transport, params: _load_model(transport, params),
    "unloadModel": lambda transport, params: unload_model(
        transport, model_id=params["modelId"]
    ),
    "getModelInfo": _request(GetModelInfoRequest, "getModelInfo", get_model_info),
    "getLoadedModelInfo": _request(
        GetLoadedModelInfoRequest, "getLoadedModelInfo", get_loaded_model_info
    ),
    # --- registry ------------------------------------------------------------
    "modelRegistryList": lambda transport, params: model_registry_list(transport),
    "modelRegistrySearch": lambda transport, params: model_registry_search(
        transport,
        filter=params.get("filter"),
        engine=params.get("engine"),
        quantization=params.get("quantization"),
        addon=params.get("addon"),
        model_type=params.get("modelType"),
    ),
    "modelRegistryGetModel": lambda transport, params: model_registry_get_model(
        transport, params["registryPath"], params["registrySource"]
    ),
    # --- runtime and host ----------------------------------------------------
    # An ergonomic wrapper, so keyword arguments -- `_request` passes a
    # validated request model positionally, which is the raw stub's convention
    # and a TypeError here.
    "cancel": lambda transport, params: cancel(
        transport,
        request_id=params.get("requestId"),
        model_id=params.get("modelId"),
        kind=params.get("kind"),
        clear_cache=params.get("clearCache"),
    ),
    "deleteCache": lambda transport, params: delete_cache(
        transport,
        all=params.get("all"),
        auto=params.get("auto"),
        kv_cache_key=params.get("kvCacheKey"),
        model_id=params.get("modelId"),
    ),
    "downloadAsset": _request(DownloadAssetRequest, "downloadAsset", download_asset),
    "getSystemResources": _request(
        GetSystemResourcesRequest, "getSystemResources", get_system_resources
    ),
    "heartbeat": _request(HeartbeatRequest, "heartbeat", heartbeat),
    "suspend": _request(SuspendRequest, "suspend", suspend),
    "resume": _request(ResumeRequest, "resume", resume),
    "state": _request(StateRequest, "state", state),
    # --- rag, vector index, finetune -----------------------------------------
    "ragIngest": _request(RagRequest, "rag", rag),
    "ragDeleteWorkspace": _request(RagRequest, "rag", rag),
    "createVectorIndex": _request(VectorIndexRequest, "vectorIndex", vector_index),
    "loadVectorIndex": _request(VectorIndexRequest, "vectorIndex", vector_index),
    "finetune": _request(FinetuneRequest, "finetune", finetune),
    # --- plugins -------------------------------------------------------------
    "invokePlugin": lambda transport, params: invoke_plugin(
        transport,
        model_id=params["modelId"],
        handler=params["handler"],
        params=params.get("params"),
    ),
}


def _snake(params: dict[str, Any]) -> dict[str, Any]:
    """camelCase step params -> the snake_case keyword arguments Python uses."""
    out: dict[str, Any] = {}
    for key, value in params.items():
        out[re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()] = value
    return out


def _load_model(transport: Any, params: dict[str, Any]) -> Any:
    async def run() -> dict[str, Any]:
        model_id = await load_model(
            transport,
            model_src=params.get("modelSrc"),
            model_type=params.get("modelType"),
            model_config=params.get("modelConfig"),
            model_name=params.get("modelName"),
            model_id=params.get("modelId"),
        )
        # JS binds `{ modelId }` so a later step can project it; matching that
        # here keeps one definition working on both clients.
        return {"modelId": model_id}

    return run()


async def _completion_stream(
    transport: Any, params: dict[str, Any], collect: str
) -> Any:
    """Fold a completion the way `collect` asks for.

    The fold is the interesting half of a streaming binding: two clients are
    only running the same test if "the text of this completion" means the same
    thing on both. Kept beside the JS fold in `tests/shared/step-bindings.ts`
    so a divergence is a one-line diff rather than an archaeology exercise.
    """
    run = completion(
        transport,
        model_id=params["modelId"],
        history=params["history"],
        stream=params.get("stream", True),
        generation_params=params.get("generationParams"),
        tools=params.get("tools"),
        response_format=params.get("responseFormat"),
        tool_dialect=params.get("toolDialect"),
    )
    if collect == "text":
        # `tool_calls` rides along with the text because a tools test needs
        # both: the model either answered or called a tool, and which one it
        # did is the question. Two folds would mean two completions.
        calls = await run.tool_calls()
        return {
            "text": await run.text(),
            "toolCalls": [
                {"name": call.name, "arguments": call.arguments} for call in calls
            ],
        }
    if collect == "events":
        return {"events": [_jsonable(event) async for event in run.events]}
    raise StepError(
        f'collect: "{collect}" is not defined for completion', incomplete=True
    )


async def _translate_stream(
    transport: Any, params: dict[str, Any], collect: str
) -> Any:
    if collect != "text":
        raise StepError(
            f'collect: "{collect}" is not defined for translate', incomplete=True
        )
    stream = params.get("stream", True)
    run = translate(
        transport,
        model_id=params["modelId"],
        text=params["text"],
        model_type=params["modelType"],
        to=params.get("to"),
        from_=params.get("from"),
        stream=stream,
        context=params.get("context"),
    )
    if not stream:
        return {"text": await run.text}
    # `text` resolves to the empty string in streaming mode on both clients, so
    # the fold has to follow the mode rather than always await the same handle.
    text = ""
    async for token in run.token_stream:
        text += token
    return {"text": text}


# Methods whose result is a stream handle rather than a value. A step reaches
# these through `collect`, which names the fold it wants.
#
# Only the methods for which the Python SDK ships a *run handle* are here. The
# generated `<name>_stream` stubs exist for the rest, but they yield raw wire
# chunks: folding those in this client would mean writing the SDK's ergonomics
# inside the test client, and the test would then pass while the SDK still had
# no wrapper. That is the one thing this whole exercise is meant to prevent, so
# those methods report `incomplete` with the reason instead -- see
# NO_RUN_HANDLE.
STREAMS: dict[str, Callable[[Any, dict[str, Any], str], Any]] = {
    "completion": _completion_stream,
    "translate": _translate_stream,
}

# Streaming methods where the Python SDK has only the generated stub.
#
# This table is the client's own roadmap, and shrinking it is the number the
# release claim is about. Each entry says what JS returns, because that is the
# shape a definition written against the reference client assumes.
NO_RUN_HANDLE: dict[str, str] = {
    "ocr": (
        "Python has only the generated ocr_stream stub, which yields raw wire "
        "chunks; JS returns a run with blockStream/blocks/stats. Needs an "
        "ergonomic wrapper before a definition written against the JS shape "
        "can run here."
    ),
    "transcribeStream": (
        "Python has transcribe_stream and transcribe_stream_session, neither "
        "shaped like the JS generator of segments. Needs an ergonomic wrapper."
    ),
    "textToSpeech": (
        "Python has only the generated text_to_speech_stream stub; JS returns "
        "a run with bufferStream/buffer/done/sampleRate. Needs an ergonomic "
        "wrapper."
    ),
    "diffusion": (
        "Python has only the generated diffusion_stream stub; JS returns a run "
        "with progressStream/outputs/stats. Needs an ergonomic wrapper."
    ),
    "upscale": (
        "Python has only the generated upscale_stream stub; JS returns a run "
        "with outputs/stats. Needs an ergonomic wrapper."
    ),
    "audioGen": (
        "Python has only the generated audio_gen_stream stub; JS returns a run "
        "with progressStream/audio/stats. Needs an ergonomic wrapper."
    ),
    "audioEdit": (
        "Python has only the generated audio_edit_stream stub; JS returns a run "
        "with progressStream/audio/stats. Needs an ergonomic wrapper."
    ),
    "audioUnderstand": (
        "Python has only the generated audio_understand stub; JS returns a run "
        "with progressStream/description/stats. Needs an ergonomic wrapper."
    ),
    "batchCompletion": (
        "Python has only the generated batch_completion_stream stub; JS returns "
        "a run with events/results. Needs an ergonomic wrapper."
    ),
    "worldStep": (
        "Python has only the generated world_step_stream stub; JS returns a run "
        "with frameStream/progressStream. Needs an ergonomic wrapper."
    ),
    "finetune": (
        "Python's finetune is the generated reply stub; JS returns a handle "
        "with progressStream/result. Needs an ergonomic wrapper for the "
        "progress-bearing form."
    ),
    "invokePluginStream": (
        "Python's invoke_plugin_stream yields decoded chunks but is not wired "
        "into these bindings yet."
    ),
}

# Methods whose Python surface is NOT yet the ergonomic equivalent of the JS
# one. The generated stub exists and the wire contract matches, but the shape
# the caller gets differs — so a definition written against the JS shape cannot
# run here yet.
#
# This is the same family as the missing stream folds: the typed surface is
# generated, the behaviour above it is hand-written per client, and that is
# exactly where a release claim needs evidence. Found by migrating a second
# category, which is what migrating one is for.
NOT_YET_ERGONOMIC: dict[str, str] = {
    "getLoadedModelInfo": (
        "Python's get_loaded_model_info returns the raw {type, info} envelope; "
        "the JS client returns info directly. Needs an ergonomic wrapper before "
        "a definition written against the JS shape can run here."
    ),
}

_INDEX = re.compile(r"^(.*?)\[(\d+)\]$")


class _Missing:
    """An optional reference that resolved to nothing.

    Distinct from None so a step can still pass an explicit null where the
    contract has one.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<missing>"


_MISSING = _Missing()


class StepError(Exception):
    """A step could not run. Carries whether that is a failure or a gap."""

    def __init__(self, message: str, incomplete: bool = False) -> None:
        super().__init__(message)
        self.incomplete = incomplete


class Interpreter:
    def __init__(self, resources: ResourceManager, log) -> None:
        self._resources = resources
        self._log = log

    async def run(
        self,
        steps: list[dict[str, Any]],
        params: dict[str, Any],
        expectation: dict[str, Any],
    ) -> StepResult:
        # Evict anything this test did not declare before it starts, so a run
        # does not depend on the order tests happened to arrive in.
        declared = {
            dep
            for step in steps
            if "useModel" in step
            for dep in step["useModel"].get("deps", [])
        }
        await self._resources.evict_all_except(declared)

        scope: dict[str, Any] = {"params": params}

        try:
            last_assert = await self._run_steps(steps, scope, expectation)
        except StepError as error:
            if error.incomplete:
                return StepResult.incomplete(str(error))
            return StepResult.fail(str(error))
        except Exception as error:  # noqa: BLE001 - any client error fails the test
            return StepResult.fail(f"{type(error).__name__}: {error}")

        if last_assert is None:
            return StepResult.fail("test body ran but asserted nothing")
        return last_assert

    async def _run_steps(
        self,
        steps: list[dict[str, Any]],
        scope: dict[str, Any],
        expectation: dict[str, Any],
    ) -> StepResult | None:
        last_assert: StepResult | None = None
        for step in steps:
            result = await self._run_step(step, scope, expectation)
            if result is None:
                continue
            # The first failing check decides the test. A migrated executor
            # often becomes several checks in a row -- shape, then length, then
            # value -- and without this a later passing one would mask an
            # earlier failure. Mirrors the JS interpreter.
            if not result.passed:
                return result
            last_assert = result
        return last_assert

    async def _run_step(
        self,
        step: dict[str, Any],
        scope: dict[str, Any],
        expectation: dict[str, Any],
    ) -> StepResult | None:
        if len(step) != 1:
            raise StepError(
                f"a step must carry exactly one operation, got {sorted(step)}"
            )
        op, body = next(iter(step.items()))

        if op == "useModel":
            return await self._use_model(body, scope)
        if op == "modelSource":
            return self._model_source(body, scope)
        if op == "asset":
            return self._asset(body, scope)
        if op == "call":
            return await self._call(body, scope)
        if op == "callError":
            return await self._call_error(body, scope)
        if op == "repeat":
            return await self._repeat(body, scope, expectation)
        if op == "project":
            return self._project(body, scope)
        if op == "assert":
            return self._assert(body, scope, expectation)

        raise StepError(
            f'step operation "{op}" is not implemented by the Python interpreter yet',
            incomplete=True,
        )

    async def _call_error(self, body: dict[str, Any], scope: dict[str, Any]) -> None:
        """A call expected to fail. Binds { code, message }.

        If the call succeeds the test fails: an error test that quietly passes
        when the error stops happening is worse than no test.
        """
        method = body["method"]
        call = CALLS.get(method)
        collect = body.get("collect")
        if call is None and not collect:
            raise StepError(
                f'SDK method "{method}" is not wired into the Python interpreter yet',
                incomplete=True,
            )

        params = self._call_params(body.get("params"), scope)

        self._log(f"callError {method}")
        try:
            await self._invoke(method, call, params, collect)
        except StepError:
            raise
        except Exception as error:  # noqa: BLE001 - the rejection is the subject
            code = getattr(error, "code", None)
            # `hasCause` and a present `code` are what an "errors are
            # structured" test asks about. Binding them here keeps that
            # question answerable without a step that reaches into a
            # language's exception object.
            scope[body["as"]] = {
                "code": "" if code is None else str(code),
                "message": str(error),
                "hasCause": error.__cause__ is not None,
            }
            return None

        raise StepError(f"{method} was expected to fail but resolved")

    async def _invoke(
        self,
        method: str,
        call: Callable[[Any, dict[str, Any]], Any] | None,
        params: dict[str, Any],
        collect: str | None,
    ) -> Any:
        """One call, folded if the step asked for a fold."""
        if collect:
            stream = STREAMS.get(method)
            if stream is None:
                raise StepError(
                    NO_RUN_HANDLE.get(method)
                    or f'SDK method "{method}" has no stream fold in the Python '
                    "interpreter yet",
                    incomplete=True,
                )
            return await stream(self._resources.transport, params, collect)
        if call is None:
            raise StepError(
                f'SDK method "{method}" is not wired into the Python interpreter yet',
                incomplete=True,
            )
        return await call(self._resources.transport, params)

    async def _repeat(
        self,
        body: dict[str, Any],
        scope: dict[str, Any],
        expectation: dict[str, Any],
    ) -> None:
        items = self._resolve(body["over"], scope)
        if not isinstance(items, list):
            raise StepError(f'repeat.over "{body["over"]}" did not resolve to a list')

        steps: list[dict[str, Any]] = body["steps"]
        collected: list[Any] = []
        for item in items:
            # Each iteration gets its own scope so a binding from one item
            # cannot leak into the next.
            inner = {**scope, body["as"]: item}
            failure = await self._run_steps(steps, inner, expectation)
            if failure is not None and not failure.passed:
                # An iteration that failed its own check must stop the repeat
                # rather than contribute a half-built value to `collectInto`.
                raise StepError(
                    failure.output or "repeat iteration failed",
                    incomplete=failure.is_incomplete,
                )
            collected.append(inner.get(_last_binding(steps) or "result"))

        scope[body["collectInto"]] = collected
        return None

    # Asset family -> the directory it lives in, mirroring ASSET_ROOTS in
    # tests/shared/step-bindings.ts. Two clients only run the same test if
    # `{ kind: "image", file: "elephant.jpg" }` means the same file on both.
    ASSET_ROOTS = {
        "image": "images",
        "audio": "audio",
        "document": "documents",
        "neural": "neural",
    }

    def _asset(self, body: dict[str, Any], scope: dict[str, Any]) -> None:
        """Resolve a bundled fixture: its bytes, or a path the SDK can open."""
        # `kind` and `file` resolve like any other value: a category whose
        # tests differ only in which fixture they use should carry one body and
        # name the file in its params.
        kind = str(self._resolve(body["kind"], scope))
        root = self.ASSET_ROOTS.get(kind)
        if root is None:
            raise StepError(
                f'asset kind "{kind}" is not known to the Python client',
                incomplete=True,
            )
        absolute = ASSET_ROOT / root / str(self._resolve(body["file"], scope))
        if not absolute.exists():
            # A missing fixture is a real failure, not a client gap.
            raise StepError(f"asset not found: {absolute}")
        scope[body["as"]] = (
            str(absolute) if body.get("form") == "path" else absolute.read_bytes()
        )
        return None

    def _model_source(self, body: dict[str, Any], scope: dict[str, Any]) -> None:
        """The model source behind a resource key, without loading it."""
        try:
            scope[body["as"]] = self._resources.source_of(
                str(self._resolve(body["dep"], scope))
            )
        except UnknownResourceError as error:
            raise StepError(str(error), incomplete=True) from error
        return None

    async def _use_model(self, body: dict[str, Any], scope: dict[str, Any]) -> None:
        deps: list[str] = body["deps"]
        try:
            model_ids = [await self._resources.ensure_loaded(dep) for dep in deps]
        except UnknownResourceError as error:
            raise StepError(str(error), incomplete=True) from error

        name = body.get("as")
        if name:
            # One key binds the id itself; several bind the list, so a step can
            # address them positionally.
            scope[name] = model_ids[0] if len(model_ids) == 1 else model_ids
        # Convention: the first declared model is addressable as $model so the
        # common single-model test needs no explicit `as`.
        scope.setdefault("model", model_ids[0])
        return None

    async def _call(self, body: dict[str, Any], scope: dict[str, Any]) -> None:
        method = body["method"]
        call = CALLS.get(method)
        collect = body.get("collect")
        # A streaming method lives in STREAMS, not CALLS, so a step that asks
        # for a fold must be allowed through even though CALLS has no entry.
        if call is None and not collect:
            raise StepError(
                f'SDK method "{method}" is not wired into the Python interpreter yet',
                incomplete=True,
            )
        gap = NOT_YET_ERGONOMIC.get(method)
        if gap:
            raise StepError(gap, incomplete=True)

        params = self._call_params(body.get("params"), scope)

        self._log(f"call {method}")
        response = await self._invoke(method, call, params, collect)

        result = _jsonable(response)
        if isinstance(result, dict) and result.get("success") is False:
            raise StepError(f"{method} failed: {result.get('error')}")

        name = body.get("as")
        scope[name or "result"] = result
        return None

    def _project(self, body: dict[str, Any], scope: dict[str, Any]) -> None:
        source = self._resolve(body["from"], scope)
        value = _walk(source, body["path"])
        join = body.get("join")
        if join is not None and isinstance(value, (list, tuple)):
            value = join.join(str(v) for v in value)
        scope[body["as"]] = value
        return None

    def _assert(
        self,
        body: dict[str, Any],
        scope: dict[str, Any],
        expectation: dict[str, Any],
    ) -> StepResult:
        value = self._resolve(body["on"], scope)
        named = body.get("named")
        if named:
            assertion = ASSERTIONS.get(named)
            if assertion is None:
                raise StepError(
                    f'named assertion "{named}" is not in the Python registry yet',
                    incomplete=True,
                )
            # `with` lets a check compare the result against something the test
            # set up, not only against a constant.
            args = self._resolve(body.get("with", {}), scope)
            result = assertion(value, args)
            result.asserted_value = value
            return result
        result = validate(value, expectation)
        result.asserted_value = value
        return result

    def _call_params(
        self, params: dict[str, Any] | None, scope: dict[str, Any]
    ) -> dict[str, Any]:
        """Resolve a call's parameters, dropping the ones that resolved to nothing.

        An optional reference that is not there must leave the argument out
        entirely, not pass it as None: an SDK that distinguishes "absent" from
        "explicitly nothing" would otherwise see a different call than the test
        meant to make, and the two clients would have to agree on which.
        """
        resolved = self._resolve(params or {}, scope)
        return {k: v for k, v in resolved.items() if v is not _MISSING}

    def _resolve(self, value: Any, scope: dict[str, Any]) -> Any:
        """Replace `$name` / `$params.x` references, recursively.

        A trailing `?` marks the reference optional: a path that is not there
        resolves to the missing marker instead of failing the step. Most calls
        in the catalog take optional arguments, and without this every test
        would have to restate its own params inside its steps just to leave one
        of them out.
        """
        if isinstance(value, str) and value.startswith("$"):
            if value.endswith("?"):
                try:
                    return _walk(scope, value[1:-1])
                except StepError:
                    return _MISSING
            return _walk(scope, value[1:])
        if isinstance(value, dict):
            return {k: self._resolve(v, scope) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve(v, scope) for v in value]
        return value


def _jsonable(value: Any) -> Any:
    """Render a response the way the JS client would report it.

    mode="json" matters: without it pydantic leaves enums and dates as Python
    objects. They would not serialise onto the bridge, and worse, the JS client
    reports plain JSON for the same field -- so the cross-client value
    comparison would report drift that is not there. Lists are mapped rather
    than dumped whole, because the ergonomic wrappers return lists of models.
    """
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", by_alias=True)
    return value


def _last_binding(steps: list[dict[str, Any]]) -> str | None:
    """Name the last value a nested step list bound, for `repeat.collectInto`.

    Mirrors the JS interpreter so both clients collect the same thing.
    """
    for step in reversed(steps):
        if "project" in step:
            return step["project"]["as"]
        if "call" in step:
            return step["call"].get("as") or "result"
        if "asset" in step:
            return step["asset"]["as"]
        if "modelSource" in step:
            return step["modelSource"]["as"]
    return None


def _walk(source: Any, path: str) -> Any:
    """Resolve a dotted path with optional [i] indexes."""
    current = source
    for segment in path.split("."):
        match = _INDEX.match(segment)
        index: int | None = None
        if match:
            segment, index = match.group(1), int(match.group(2))
        if segment:
            # JS's walk() guards this and raises its own error, which the
            # optional-reference handler then swallows. Without the same guard
            # here a null intermediate raises AttributeError, which that
            # handler does not catch -- so `$a.b?` resolves to nothing on JS
            # and fails the whole test on Python.
            if current is None:
                raise StepError(f'path "{path}" walked off a null at "{segment}"')
            if isinstance(current, dict):
                if segment not in current:
                    raise StepError(f'path "{path}" has no "{segment}"')
                current = current[segment]
            else:
                try:
                    current = getattr(current, segment)
                except AttributeError as error:
                    raise StepError(f'path "{path}" has no "{segment}"') from error
        if index is not None:
            current = current[index]
    return current
