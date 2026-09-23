import * as fs from 'node:fs'
import * as path from 'node:path'
import {
  audioEdit,
  audioGen,
  audioUnderstand,
  batchCompletion,
  bciTranscribe,
  cancel,
  classify,
  completion,
  createVectorIndex,
  deleteCache,
  diffusion,
  downloadAsset,
  embed,
  finetune,
  getLoadedModelInfo,
  getModelInfo,
  getSystemResources,
  heartbeat,
  invokePlugin,
  invokePluginStream,
  loadModel,
  loadVectorIndex,
  modelRegistryGetModel,
  modelRegistryList,
  modelRegistrySearch,
  ocr,
  ragCloseWorkspace,
  ragDeleteWorkspace,
  ragIngest,
  resume,
  state,
  suspend,
  textToSpeech,
  transcribe,
  transcribeStream,
  translate,
  unloadModel,
  upscale,
  vla,
  vlaHparams,
  vlaSetEmbodiment,
  worldCreateScene,
  worldStep,
  AUDIOGEN_INPUT_SAMPLE_RATE,
  AUDIOGEN_INPUT_CHANNELS
} from '@qvac/sdk'
import { StepIncompleteError, type CollectMode, type StepBindings } from '@qvac/test-suite'
import type { ResourceManager } from './resource-manager.js'

/**
 * How the shared step interpreter reaches this SDK.
 *
 * The framework knows nothing about `@qvac/sdk`; it asks these bindings to
 * load a model behind a resource key and to make a call. Everything else —
 * reference resolution, ordering, what counts as pass / fail / incomplete —
 * lives in the interpreter, once, and is therefore identical on every client.
 *
 * Calls go through the public SDK surface on purpose. A binding that reached
 * past it into the worker would prove only that the engine works, which is not
 * the question: the question is whether two clients driving the same engine
 * agree.
 */

/**
 * Contract method name -> the SDK function that serves it.
 *
 * Deliberately explicit rather than reflective: a typo in a step should be a
 * clear `incomplete` from the interpreter, not a mystery at call time. Grows
 * one entry at a time as categories migrate.
 */
const CALLS: Record<string, (params: never) => Promise<unknown>> = {
  // --- inference, request/reply -------------------------------------------
  embed: (params) => embed(params),
  classify: async (params) => ({ results: await classify(params) }),
  transcribe: async (params) => ({ text: await transcribe(params) }),
  bciTranscribe: async (params) => ({ text: await bciTranscribe(params) }),
  vla: (params) => vla(params),
  vlaHparams: (params) => vlaHparams(params),
  vlaSetEmbodiment: (params) => vlaSetEmbodiment(params),

  // --- models --------------------------------------------------------------
  loadModel: async (params) => ({ modelId: await loadModel(params) }),
  unloadModel: (params) => unloadModel(params),
  getModelInfo: (params) => getModelInfo(params),
  getLoadedModelInfo: (params) => getLoadedModelInfo(params),

  // --- registry ------------------------------------------------------------
  // The trio takes its arguments differently in each language -- positional
  // here, keyword in Python. The contract name and the params object in the
  // step are what both sides agree on; adapting to the local signature is
  // precisely what a binding is for.
  modelRegistryList: () => modelRegistryList(),
  modelRegistrySearch: (params) => modelRegistrySearch(params),
  modelRegistryGetModel: (params: never) => {
    const p = params as unknown as { registryPath: string; registrySource: string }
    return modelRegistryGetModel(p.registryPath, p.registrySource)
  },

  // --- runtime and host ----------------------------------------------------
  cancel: (params) => cancel(params),
  deleteCache: (params) => deleteCache(params),
  downloadAsset: async (params) => ({ path: await downloadAsset(params) }),
  getSystemResources: (params) => getSystemResources(params),
  heartbeat: () => heartbeat(),
  suspend: async () => {
    await suspend()
    return { suspended: true }
  },
  resume: async () => {
    await resume()
    return { resumed: true }
  },
  state: async () => ({ state: await state() }),

  // --- rag and vector index ------------------------------------------------
  ragIngest: (params) => ragIngest(params),
  ragCloseWorkspace: async (params) => {
    await ragCloseWorkspace(params)
    return { closed: true }
  },
  ragDeleteWorkspace: async (params) => {
    await ragDeleteWorkspace(params)
    return { deleted: true }
  },
  createVectorIndex: (params) => createVectorIndex(params),
  loadVectorIndex: (params) => loadVectorIndex(params),

  // --- plugins -------------------------------------------------------------
  invokePlugin: async (params) => ({ result: await invokePlugin(params) }),

  // --- world ---------------------------------------------------------------
  worldCreateScene: async (params: never) => {
    const scene = worldCreateScene(params)
    return { requestId: scene.requestId, stats: await scene.stats }
  }
}

/** Every value of an async iterator, in order. */
async function drain<T>(source: AsyncIterable<T>): Promise<T[]> {
  const out: T[] = []
  for await (const value of source) out.push(value)
  return out
}

/** Concatenate the pieces of a text stream. */
async function joinStream(source: AsyncIterable<string>): Promise<string> {
  let text = ''
  for await (const piece of source) text += piece
  return text
}

/**
 * How a streaming call is folded into one value, per method.
 *
 * The fold is the interesting half of a streaming binding: two clients are
 * only running the same test if "the text of this completion" or "the frames
 * of this block" means the same thing on both. Each entry returns an object so
 * a later `project` step has a field to pull, and the field is named after the
 * fold -- `text`, `blocks`, `events`, `pcm`, `last`, `all` -- so a definition
 * reads the same regardless of which SDK produced the value.
 *
 * A method that has no entry for the mode a step asked for reports
 * `incomplete`. That is a gap in these bindings, not a failing test.
 */
type Fold = (params: never, collect: CollectMode) => Promise<unknown>

const STREAMS: Record<string, Fold> = {
  completion: async (params, collect) => {
    const run = completion(params)
    // `toolCalls` rides along with `text` because a tools test needs both: the
    // model either answered or called a tool, and which one it did is the
    // question. Splitting them across two folds would mean running the
    // completion twice.
    if (collect === 'text') {
      return { text: await run.text, toolCalls: (await run.toolCalls) ?? [] }
    }
    if (collect === 'events') return { events: await drain(run.events) }
    if (collect === 'all') return { all: await drain(run.tokenStream) }
    throw unsupported('completion', collect)
  },

  translate: async (params, collect) => {
    const p = params as unknown as { stream?: boolean }
    const run = translate(params)
    // `all` carries the joined text beside the tokens: "it streamed, and this
    // is what it said" is one question, and a second fold would be a second
    // translation.
    if (collect === 'all') {
      const all = await drain(run.tokenStream)
      return { all, text: all.join(''), stats: await run.stats }
    }
    if (collect !== 'text') throw unsupported('translate', collect)
    // `text` resolves to the empty string in streaming mode on both clients, so
    // the fold has to follow the mode rather than always await the same handle.
    if (p.stream) return { text: await joinStream(run.tokenStream), stats: await run.stats }
    // `translations` rides along: a batch asks about the entries and about the
    // text they join to, and a second fold would be a second translation.
    return { text: await run.text, translations: await run.translations, stats: await run.stats }
  },

  transcribeStream: async (params, collect) => {
    const pieces = await drain(transcribeStream(params) as AsyncIterable<unknown>)
    if (collect === 'blocks') return { blocks: pieces }
    if (collect === 'all') return { all: pieces }
    if (collect === 'last') return { last: pieces.at(-1) }
    if (collect === 'text') return { text: pieces.map((p) => String(p)).join('') }
    throw unsupported('transcribeStream', collect)
  },

  ocr: async (params, collect) => {
    const p = params as unknown as { stream?: boolean }
    const run = ocr(params)

    // `blocks` resolves empty in streaming mode, the same trap `translate`
    // has: the fold must follow the call's own mode rather than always await
    // the same handle, or a streaming test asserts on nothing and says so
    // only if the expectation happens to be strict.
    const blocks = p.stream ? (await drain(run.blockStream)).flat() : await run.blocks

    // `stats` rides along with every fold: a test that checks timing asks for
    // it from the same run, and a second call would time a different one.
    const stats = await run.stats

    if (collect === 'blocks') return { blocks, stats }
    if (collect === 'all' || collect === 'events') return { all: blocks, events: blocks, stats }
    if (collect === 'text') {
      // Space, not newline: this is what the executors joined with, and a
      // migrated test has to reproduce what its executor produced or it is not
      // migrated.
      return { text: blocks.map((block) => block.text).join(' '), stats }
    }
    throw unsupported('ocr', collect)
  },

  textToSpeech: async (params, collect) => {
    const run = textToSpeech(params)
    if (collect === 'pcm') {
      const pcm = await run.buffer
      return { pcm, sampleRate: await run.sampleRate, done: await run.done }
    }
    if (collect === 'all') return { all: await drain(run.bufferStream) }
    if (collect === 'events') {
      return { events: run.chunkUpdates ? await drain(run.chunkUpdates) : [] }
    }
    throw unsupported('textToSpeech', collect)
  },

  diffusion: async (params, collect) => {
    const run = diffusion(params)
    if (collect === 'events') {
      // Draining progress before awaiting the outputs is not a style choice:
      // the generator is the live side of the same stream, and awaiting first
      // would leave nothing to iterate.
      const events = await drain(run.progressStream)
      await run.outputs
      return { events }
    }
    if (collect === 'all') return { all: await run.outputs }
    if (collect === 'last') {
      const outputs = await run.outputs
      return { last: outputs.at(-1) }
    }
    throw unsupported('diffusion', collect)
  },

  upscale: async (params, collect) => {
    const run = upscale(params)
    if (collect === 'all') return { all: await run.outputs }
    if (collect === 'last') {
      const outputs = await run.outputs
      return { last: outputs.at(-1) }
    }
    throw unsupported('upscale', collect)
  },

  audioGen: async (params, collect) => audioRun(audioGen(params), collect, 'audioGen'),
  audioEdit: async (params, collect) => audioRun(audioEdit(params), collect, 'audioEdit'),

  audioUnderstand: async (params, collect) => {
    const run = audioUnderstand(params)
    if (collect === 'text') return { text: await run.description }
    if (collect === 'events') return { events: await drain(run.progressStream) }
    throw unsupported('audioUnderstand', collect)
  },

  batchCompletion: async (params, collect) => {
    const run = batchCompletion(params)
    if (collect === 'all') {
      // Events are drained alongside the results: a streaming batch test asks
      // whether every prompt produced deltas *and* whether its final agrees
      // with them, and a second fold would be a second batch.
      //
      // Both are awaited through one `Promise.all` rather than in sequence.
      // An empty batch rejects both, and awaiting them one after the other
      // leaves the second rejection with nobody listening -- which takes the
      // whole consumer process down on a test that had already passed.
      const [all, events] = await Promise.all([run.results, drain(run.events)])
      return { all, events }
    }
    if (collect === 'events') return { events: await drain(run.events) }
    throw unsupported('batchCompletion', collect)
  },

  worldStep: async (params, collect) => {
    const run = worldStep(params)
    // `frames` is the same array the generator would hand back one at a time,
    // filled by the run's own pump whether or not anyone iterates. Draining the
    // generator instead would double-buffer every image frame to arrive at the
    // identical value.
    if (collect === 'all') return { all: await run.frames }
    if (collect === 'last') {
      const frames = await run.frames
      return { last: frames.at(-1), frameCount: frames.length }
    }
    if (collect === 'events') {
      const frames = await run.frames
      return { events: await drain(run.progressStream), frameCount: frames.length }
    }
    throw unsupported('worldStep', collect)
  },

  finetune: async (params, collect) => {
    const handle = finetune(params)
    if (collect === 'events') {
      const events = await drain(handle.progressStream)
      return { events, last: await handle.result }
    }
    if (collect === 'last') return { last: await handle.result }
    throw unsupported('finetune', collect)
  },

  invokePluginStream: async (params, collect) => {
    const chunks = await drain(invokePluginStream(params))
    if (collect === 'all') return { all: chunks }
    if (collect === 'last') return { last: chunks.at(-1) }
    if (collect === 'text') return { text: chunks.map((c) => String(c)).join('') }
    throw unsupported('invokePluginStream', collect)
  }
}

/** `audioGen` and `audioEdit` return the same handle, so they fold the same. */
async function audioRun(
  run: {
    audio: Promise<unknown>
    progressStream: AsyncIterable<unknown>
    stats: Promise<unknown>
  },
  collect: CollectMode,
  method: string
): Promise<unknown> {
  if (collect === 'pcm') {
    // Progress is drained alongside the audio rather than in a second fold:
    // these tests ask whether one run produced audio *and* reported progress,
    // and a second `collect` would be a second generation -- minutes of work
    // answering a question about the first one.
    // One `Promise.all`, not three awaits in a row: a run that rejects rejects
    // all of them, and a rejection awaited second has nobody listening when
    // the first one throws -- an unhandled rejection that ends the consumer.
    const [audio, stats, events] = (await Promise.all([
      run.audio,
      run.stats,
      drain(run.progressStream)
    ])) as [
      { pcm: unknown; sampleRate: unknown; channels: unknown; bitsPerSample: unknown },
      unknown,
      unknown[]
    ]
    // Spelled out rather than passed through: Python's run hands back the same
    // four values under `data`, so naming them here is what makes
    // `$run.audio.pcm` one thing in both clients.
    return {
      audio: {
        pcm: audio.pcm,
        sampleRate: audio.sampleRate,
        channels: audio.channels,
        bitsPerSample: audio.bitsPerSample
      },
      stats,
      events
    }
  }
  if (collect === 'events') {
    const events = await drain(run.progressStream)
    await run.audio
    return { events }
  }
  throw unsupported(method, collect)
}

function unsupported(method: string, collect: CollectMode): StepIncompleteError {
  return new StepIncompleteError(`collect: "${collect}" is not defined for ${method}`)
}

/**
 * Where a test asset lives on this platform.
 *
 * The whole reason the OCR tests have two executors today — one under `node/`,
 * one under `mobile/` — is nothing but this: a filesystem path here, a
 * bundled-asset URI there. Once resolution is a step the interpreter performs,
 * that split collapses.
 *
 * Desktop and Electron read from the checkout; a mobile binding would resolve
 * the same `kind`/`file` pair through Metro instead, and the definition would
 * not change.
 */
/**
 * A fixture the catalog names but no file holds: `"2s-440hz"` is two seconds of
 * a 440 Hz tone.
 *
 * The audio tests feed a synthesized tone rather than a recording because the
 * point is a known signal, not a performance. Generating it from the name
 * keeps it a fixture both clients resolve identically -- checking in a wav
 * would work too, but then "the same source audio" would rest on a binary
 * nobody reads.
 */
const TONE = /^(\d+(?:\.\d+)?)s-(\d+(?:\.\d+)?)hz$/

/** Raw interleaved stereo 48 kHz Float32 LE PCM, the form AudioGen accepts. */
const synthesizeTone = (spec: string): Uint8Array => {
  const match = TONE.exec(spec)
  if (!match) throw new Error(`tone "${spec}" is not "<seconds>s-<frequency>hz"`)
  const [seconds, frequency] = [Number(match[1]), Number(match[2])]
  const frames = Math.round(AUDIOGEN_INPUT_SAMPLE_RATE * seconds)
  const pcm = new Float32Array(frames * AUDIOGEN_INPUT_CHANNELS)
  for (let frame = 0; frame < frames; frame++) {
    const sample = 0.1 * Math.sin((2 * Math.PI * frequency * frame) / AUDIOGEN_INPUT_SAMPLE_RATE)
    for (let channel = 0; channel < AUDIOGEN_INPUT_CHANNELS; channel++) {
      pcm[frame * AUDIOGEN_INPUT_CHANNELS + channel] = sample
    }
  }
  return new Uint8Array(pcm.buffer, pcm.byteOffset, pcm.byteLength)
}

/**
 * A fixture spelled out in hex: `"00010203"` is four bytes.
 *
 * For the deliberately malformed inputs -- four bytes that cannot be a JPEG,
 * a truncated header. Checking such a file in would hide what makes it invalid
 * behind a binary; written in the catalog, the test says it.
 */
const synthesizeBytes = (spec: string): Uint8Array => {
  if (spec.length % 2 !== 0 || !/^[0-9a-fA-F]*$/.test(spec)) {
    throw new Error(`bytes "${spec}" is not an even-length hex string`)
  }
  const out = new Uint8Array(spec.length / 2)
  for (let i = 0; i < out.length; i++) out[i] = Number.parseInt(spec.slice(i * 2, i * 2 + 2), 16)
  return out
}

const ASSET_ROOTS: Record<string, string> = {
  image: 'assets/images',
  audio: 'assets/audio',
  document: 'assets/documents',
  neural: 'assets/neural'
}

/** How much data a value carries, whether it arrived as bytes or a list. */
const byteLength = (value: unknown): number => {
  if (ArrayBuffer.isView(value)) return (value as Uint8Array).byteLength
  if (Array.isArray(value)) return value.length
  return 0
}

const asBytes = (value: unknown): Uint8Array => {
  if (value instanceof Uint8Array) return value
  if (ArrayBuffer.isView(value)) {
    const view = value as ArrayBufferView
    return new Uint8Array(view.buffer, view.byteOffset, view.byteLength)
  }
  if (Array.isArray(value)) return Uint8Array.from(value.map((item) => Number(item) & 0xff))
  return new Uint8Array()
}

const sameBytes = (left: unknown, right: unknown): boolean => {
  const a = asBytes(left)
  const b = asBytes(right)
  if (a.byteLength !== b.byteLength) return false
  for (let i = 0; i < a.byteLength; i++) if (a[i] !== b[i]) return false
  return true
}

/**
 * Checks that take two bound values rather than one.
 *
 * Distinct from the assertions because the question is about the
 * relationship: the same call made twice with one parameter changed, and the
 * claim is that the results differ -- or do not.
 */
const COMPARISONS: Record<
  string,
  (
    left: unknown,
    right: unknown,
    args: Record<string, unknown>
  ) => {
    passed: boolean
    output: string
  }
> = {
  /**
   * The two runs produced exactly the same data.
   *
   * The determinism half of a conditioning test: the same inputs twice have to
   * give the same output before "changing this one input changed the output"
   * means anything.
   */
  identicalBytes(left, right) {
    if (!sameBytes(left, right)) {
      return {
        passed: false,
        output: `expected identical output, got ${byteLength(left)} and ${byteLength(right)} byte(s) that differ`
      }
    }
    return { passed: true, output: `identical, ${byteLength(left)} byte(s)` }
  },

  /**
   * The two runs produced different data, and both produced some.
   *
   * Both halves matter: two empty results are trivially different, and a
   * conditioning test that accepted them would pass against a silent engine.
   */
  differentBytes(left, right) {
    if (byteLength(left) === 0 || byteLength(right) === 0) {
      return {
        passed: false,
        output: `one side produced nothing (${byteLength(left)} and ${byteLength(right)} byte(s))`
      }
    }
    if (sameBytes(left, right)) {
      return { passed: false, output: 'expected the outputs to differ, they are identical' }
    }
    return {
      passed: true,
      output: `differ, ${byteLength(left)} vs ${byteLength(right)} byte(s)`
    }
  },

  /**
   * The left value carries at least this many times the data of the right.
   *
   * The strongest claim available about an output sample rate: the rate itself
   * is not exposed through the public result, but a native-rate run has to
   * produce proportionally more samples than a downsampled one.
   */
  lengthRatioAtLeast(left, right, args) {
    const [leftSize, rightSize] = [byteLength(left), byteLength(right)]
    if (leftSize === 0 || rightSize === 0) {
      return {
        passed: false,
        output: `comparison produced empty output (${leftSize} and ${rightSize})`
      }
    }
    const minimum = Number(args.ratio ?? 1)
    const ratio = leftSize / rightSize
    if (ratio < minimum) {
      return {
        passed: false,
        output: `ratio too low: ${ratio.toFixed(2)} < ${minimum} (${leftSize} vs ${rightSize})`
      }
    }
    return { passed: true, output: `ratio ${ratio.toFixed(2)} (${leftSize} vs ${rightSize})` }
  }
}

/**
 * Named assertions: the checks that are more than "contains this string".
 *
 * These replace the JavaScript-function expectations in the catalog, which
 * cannot cross the wire. The name is part of the shared vocabulary, so two
 * clients checking `loadedModelInfoShape` check the same thing rather than
 * each their own idea of it — which is the difference between a shared catalog
 * that means something and one that only looks shared.
 *
 * `args` is the step's `with` block, already reference-resolved: that is what
 * lets a check compare the result against something the test set up.
 */
const ASSERTIONS: Record<
  string,
  (value: unknown, args: Record<string, unknown>) => { passed: boolean; output: string }
> = {
  /**
   * The collection has exactly the expected number of elements.
   *
   * Deliberately generic: `topK: 1 must truncate to one result` is the same
   * check as "this many blocks came back", and a registry of one-off names
   * would defeat the point of a shared vocabulary.
   */
  lengthIs(value, args) {
    if (!Array.isArray(value)) {
      return { passed: false, output: `expected an array, got ${typeof value}` }
    }
    const expected = Number(args.length)
    if (value.length !== expected) {
      return { passed: false, output: `expected ${expected} element(s), got ${value.length}` }
    }
    return { passed: true, output: `${value.length} element(s)` }
  },

  /**
   * The collection has at least this many elements.
   *
   * The floor half of `lengthIs`: "the registry lists models" and "more than
   * one result came back" are the same check with a different bound, and a
   * test that pinned the exact count would fail whenever the registry grew.
   */
  lengthAtLeast(value, args) {
    if (!Array.isArray(value)) {
      return { passed: false, output: `expected an array, got ${typeof value}` }
    }
    const minimum = Number(args.length)
    if (value.length < minimum) {
      return { passed: false, output: `expected at least ${minimum}, got ${value.length}` }
    }
    return { passed: true, output: `${value.length} element(s)` }
  },

  /**
   * Every element's named field sits inside the range.
   *
   * A probability is in [0,1] and a utilisation is in [0,1]; naming the bound
   * in the test rather than the registry keeps one check answering both.
   */
  numbersInRange(value, args) {
    const items = (Array.isArray(value) ? value : []) as Array<Record<string, unknown>>
    const field = String(args.field)
    const min = Number(args.min)
    const max = Number(args.max)
    for (const item of items) {
      const measured = item[field]
      if (typeof measured !== 'number' || measured < min || measured > max) {
        return {
          passed: false,
          output: `${field} is outside [${min}, ${max}]: ${JSON.stringify(measured)}`
        }
      }
    }
    return { passed: true, output: `${items.length} value(s) within [${min}, ${max}]` }
  },

  /**
   * The value has no text in it.
   *
   * The empty-input tests: an empty prompt has nothing to translate, and the
   * claim is that the client says so rather than inventing output.
   */
  isEmptyText(value) {
    const text = typeof value === 'string' ? value : ''
    if (value !== undefined && value !== null && typeof value !== 'string') {
      return { passed: false, output: `expected a string, got ${typeof value}` }
    }
    return text.trim().length === 0
      ? { passed: true, output: '(empty)' }
      : { passed: false, output: `expected no text, got: ${text.slice(0, 120)}` }
  },

  /** The list is ordered by the named field, largest first. */
  sortedDescendingBy(value, args) {
    const items = (Array.isArray(value) ? value : []) as Array<Record<string, unknown>>
    const field = String(args.field)
    for (let i = 1; i < items.length; i++) {
      const previous = Number(items[i - 1]?.[field] ?? 0)
      const current = Number(items[i]?.[field] ?? 0)
      if (current > previous) {
        return { passed: false, output: `not sorted by ${field} at index ${i}` }
      }
    }
    return { passed: true, output: `${items.length} element(s) in order` }
  },

  /**
   * The named field sums to a value, within a tolerance.
   *
   * Softmax probabilities sum to one; the tolerance is what keeps that a claim
   * about the model rather than about float accumulation order.
   */
  sumsTo(value, args) {
    const items = (Array.isArray(value) ? value : []) as Array<Record<string, unknown>>
    const field = String(args.field)
    const total = items.reduce((sum, item) => sum + Number(item[field] ?? 0), 0)
    const expected = Number(args.total)
    const tolerance = Number(args.tolerance ?? 1e-3)
    if (Math.abs(total - expected) > tolerance) {
      return {
        passed: false,
        output: `${field} sums to ${total}, not within ${tolerance} of ${expected}`
      }
    }
    return { passed: true, output: `${field} sums to ${total}` }
  },

  /**
   * A `getSystemResources` record is well formed.
   *
   * Three claims the executor made inline, kept together because they are one
   * question about one record: every metric that reports `supported` says
   * where the number came from and none that does not report a value anyway;
   * a GPU is identified by an opaque id and never by the raw vendor/device
   * identifiers, which is a privacy boundary rather than a shape detail; and
   * a requested sample correlates with the capabilities it was taken against.
   *
   * `sample` says whether one was asked for -- a sample that arrives
   * unrequested is as much a failure as one that is missing.
   */
  systemResourcesShape(value, args) {
    const problems: string[] = []
    const record = (value ?? {}) as {
      capabilities?: Record<string, never>
      sample?: Record<string, never>
    }

    type Metric = { status?: string; value?: unknown; provenance?: { source?: string } }
    const metric = (m: unknown, label: string): Metric => {
      const measured = (m ?? {}) as Metric
      if (measured.status === 'supported') {
        if (!measured.provenance?.source) problems.push(`${label} has no provenance source`)
      } else if ('value' in measured) {
        problems.push(`${label} exposes a value with status ${String(measured.status)}`)
      }
      return measured
    }
    const utilization = (m: unknown, label: string) => {
      const measured = metric(m, label)
      const reading = measured.value as number
      if (measured.status === 'supported' && (reading < 0 || reading > 1)) {
        problems.push(`${label} is outside 0..1: ${reading}`)
      }
    }

    const capabilities = (record.capabilities ?? {}) as Record<string, never>
    if (!record.capabilities) problems.push('capabilities are missing')
    metric(capabilities['cpu'], 'capabilities.cpu')
    metric((capabilities['memory'] ?? {})['totalBytes'], 'capabilities.memory.totalBytes')
    const capabilityGpus = metric(capabilities['gpus'], 'capabilities.gpus')

    const RAW_IDENTITY = ['vendorId', 'deviceId', 'subsystemId', 'revision']
    let capabilityIds: string[] | undefined
    if (capabilityGpus.status === 'supported') {
      const gpus = (capabilityGpus.value ?? []) as Array<Record<string, unknown>>
      capabilityIds = gpus.map((gpu) => String(gpu['id']))
      for (const gpu of gpus) {
        if (!gpu['id']) problems.push('GPU has no opaque ID')
        for (const field of RAW_IDENTITY) {
          if (field in gpu) problems.push(`GPU exposes private identity field ${field}`)
        }
      }
    }

    if (!args.sample) {
      if (record.sample) problems.push('sample returned when it was not requested')
      return problems.length > 0
        ? { passed: false, output: problems.join('; ') }
        : { passed: true, output: 'capabilities valid; sample omitted' }
    }

    const sample = (record.sample ?? {}) as Record<string, never>
    if (!record.sample) {
      problems.push('requested sample is missing')
      return { passed: false, output: problems.join('; ') }
    }
    utilization(sample['cpu'], 'sample.cpu')
    const memory = (sample['memory'] ?? {}) as Record<string, never>
    metric(memory['usedBytes'], 'sample.memory.usedBytes')
    metric(memory['totalBytes'], 'sample.memory.totalBytes')
    metric(memory['processUsedBytes'], 'sample.memory.processUsedBytes')
    const allowance = metric(memory['processAvailableBytes'], 'sample.memory.processAvailableBytes')
    if (allowance.status === 'supported') {
      const reading = allowance.value as number
      if (reading <= 0) {
        problems.push(`sample.memory.processAvailableBytes is not positive: ${reading}`)
      }
      const scope = (allowance.provenance as { scope?: string } | undefined)?.scope
      if (scope !== 'process') {
        problems.push(`sample.memory.processAvailableBytes carries scope ${String(scope)}`)
      }
    } else if (args.platform === 'ios') {
      problems.push(`sample.memory.processAvailableBytes is ${String(allowance.status)} on iOS`)
    }

    const sampleGpus = metric(sample['gpus'], 'sample.gpus')
    if (capabilityIds && sampleGpus.status === 'supported') {
      const gpus = (sampleGpus.value ?? []) as Array<Record<string, unknown>>
      if (capabilityIds.join(',') !== gpus.map((gpu) => String(gpu['id'])).join(',')) {
        problems.push('capability and sample GPU IDs do not correlate')
      }
      for (const gpu of gpus) {
        utilization(gpu['compute'], `sample.gpus.${String(gpu['id'])}.compute`)
        utilization(gpu['encode'], `sample.gpus.${String(gpu['id'])}.encode`)
        utilization(gpu['decode'], `sample.gpus.${String(gpu['id'])}.decode`)
      }
    }

    return problems.length > 0
      ? { passed: false, output: problems.join('; ') }
      : { passed: true, output: 'capabilities valid; sample valid' }
  },

  /**
   * Each named result's text carries what that result was asked for.
   *
   * A batch answers several prompts at once, so "the output contains both
   * markers" is not the question -- either prompt could have produced both.
   * `mode: 'any'` is the looser form a vision prompt needs, where several
   * words would each be a right answer.
   */
  textsById(value, args) {
    const results = (Array.isArray(value) ? value : []) as Array<{
      id?: string
      final?: { contentText?: string }
    }>
    const byId = new Map(results.map((result) => [result.id, result.final?.contentText ?? '']))
    const expected = (args.expect ?? {}) as Record<string, string[]>
    const any = args.mode === 'any'

    for (const [id, terms] of Object.entries(expected)) {
      const text = byId.get(id)
      if (text === undefined) {
        return { passed: false, output: `no result for id "${id}": got ${[...byId.keys()]}` }
      }
      const lower = text.toLowerCase()
      const hits = terms.filter((term) => lower.includes(term.toLowerCase()))
      if (any ? hits.length === 0 : hits.length !== terms.length) {
        const missing = terms.filter((term) => !hits.includes(term))
        return {
          passed: false,
          output: `"${id}" is missing ${any ? 'any of' : ''} ${JSON.stringify(missing)}: ${text.slice(0, 160)}`
        }
      }
    }
    return { passed: true, output: `${Object.keys(expected).length} id(s) matched` }
  },

  /**
   * Every named result was streamed, not just delivered.
   *
   * The batch's final results look the same whether the text arrived in one
   * frame or in fifty, so a streaming test that only read the finals would
   * pass with streaming switched off.
   */
  streamedEachId(value, args) {
    const events = (Array.isArray(value) ? value : []) as Array<{
      id?: string
      event?: { type?: string; text?: string }
    }>
    const counts = new Map<string, number>()
    for (const { id, event } of events) {
      if (id === undefined) continue
      if (event?.type === 'contentDelta' && (event.text ?? '').length > 0) {
        counts.set(id, (counts.get(id) ?? 0) + 1)
      }
    }
    const missing = ((args.ids ?? []) as string[]).filter((id) => (counts.get(id) ?? 0) === 0)
    if (missing.length > 0) {
      return { passed: false, output: `no streamed content for: ${missing.join(', ')}` }
    }
    return { passed: true, output: `streamed ${[...counts.values()].join('/')} delta(s)` }
  },

  /**
   * These results made no tool call.
   *
   * The other half of `toolCallShape`: a batch where one prompt declares a
   * tool and another does not is only answered if the second one stayed quiet.
   */
  noToolCallsFor(value, args) {
    const results = (Array.isArray(value) ? value : []) as Array<{
      id?: string
      final?: { toolCalls?: Array<{ name?: string }> }
    }>
    for (const id of (args.ids ?? []) as string[]) {
      const calls = results.find((result) => result.id === id)?.final?.toolCalls ?? []
      if (calls.length > 0) {
        return {
          passed: false,
          output: `"${id}" was expected to make no tool call, made: ${calls.map((c) => c.name).join(', ')}`
        }
      }
    }
    return { passed: true, output: `${(args.ids as string[]).length} id(s) stayed quiet` }
  },

  /**
   * Every named field is present on the value.
   *
   * Replaces the inline "which required fields are missing" loops that several
   * executors grew independently. Generic on purpose: the field list belongs to
   * the test, not to the assertion registry.
   */
  fieldsPresent(value, args) {
    if (!value || typeof value !== 'object') {
      return { passed: false, output: `expected an object, got ${typeof value}` }
    }
    const record = value as Record<string, unknown>
    const fields = (args.fields ?? []) as string[]
    const missing = fields.filter((field) => record[field] === undefined)
    if (missing.length > 0) {
      return { passed: false, output: `missing fields: ${missing.join(', ')}` }
    }
    return { passed: true, output: `${fields.length} field(s) present` }
  },

  /**
   * The text is exactly the parts joined by the separator.
   *
   * A batch translation returns both the entries and one text, and the claim
   * is that they are the same answer in two shapes rather than two answers.
   */
  equalsJoined(value, args) {
    const parts = ((args.parts ?? []) as unknown[]).map(String)
    const expected = parts.join(String(args.separator ?? '\n'))
    if (value !== expected) {
      return {
        passed: false,
        output: `expected ${JSON.stringify(expected)}, got ${JSON.stringify(value)}`
      }
    }
    return { passed: true, output: `${parts.length} part(s) joined` }
  },

  /**
   * At least one of the named fields is present.
   *
   * For the readings an engine may report in more than one shape: which timing
   * field a backend fills is its business, that it reported timing at all is
   * the claim.
   */
  anyFieldPresent(value, args) {
    const record = (value ?? {}) as Record<string, unknown>
    const fields = (args.fields ?? []) as string[]
    const found = fields.filter((field) => record[field] !== undefined)
    if (found.length === 0) {
      return { passed: false, output: `none of ${fields.join(', ')} are present` }
    }
    return { passed: true, output: `${found.join(', ')} present` }
  },

  /**
   * Two records agree on the named fields.
   *
   * Compared as strings so a client that returns a number where another
   * returns a numeric string is not reported as drift -- the question here is
   * whether two views of the same record agree, not how each typed it.
   */
  fieldsMatch(value, args) {
    const left = (value ?? {}) as Record<string, unknown>
    const right = (args.expected ?? {}) as Record<string, unknown>
    const fields = (args.fields ?? []) as string[]
    const mismatched = fields.filter((field) => String(left[field]) !== String(right[field]))
    if (mismatched.length > 0) {
      return {
        passed: false,
        output: mismatched
          .map((field) => `${field}: ${String(left[field])} != ${String(right[field])}`)
          .join('; ')
      }
    }
    return { passed: true, output: `${fields.length} field(s) match` }
  },

  /**
   * The rejection carried machine-readable structure, not just a string.
   *
   * A chained cause or a present error code both answer that; which one a
   * given SDK surfaces is an implementation choice, and pinning the test to
   * one of them would make it a test of that choice rather than of the
   * guarantee.
   */
  errorIsStructured(value) {
    const err = (value ?? {}) as { code?: string; hasCause?: boolean; message?: string }
    const hasCode = typeof err.code === 'string' && err.code.length > 0
    if (!hasCode && !err.hasCause) {
      return {
        passed: false,
        output: `rejection carried neither a code nor a cause: ${err.message ?? '(no message)'}`
      }
    }
    return {
      passed: true,
      output: `hasCause=${Boolean(err.hasCause)}, code=${err.code || '(none)'}`
    }
  },

  /**
   * The value is a string with something in it.
   *
   * `expectedType: 'string'` only asks about the type, and `minLength` in the
   * expectation applies to arrays, so "it produced text" had no way to be said
   * until now. Every generative category needs it.
   */
  nonEmptyText(value) {
    if (typeof value !== 'string') {
      return { passed: false, output: `expected a string, got ${typeof value}` }
    }
    if (value.trim().length === 0) {
      return { passed: false, output: 'expected text, got an empty string' }
    }
    return { passed: true, output: `${value.length} character(s)` }
  },

  /**
   * The rejection is the one the test meant, by code and by wording.
   *
   * `messageNotMatching` is the half that is easy to forget and the reason
   * this is not just a `contains`: several error tests exist to prove a bad
   * argument is rejected *by the SDK* rather than forwarded to the addon, and
   * only the wording of the failure tells those two apart.
   */
  errorMatches(value, args) {
    const err = (value ?? {}) as { code?: string; message?: string }
    const message = err.message ?? ''

    if (args.code !== undefined && String(err.code) !== String(args.code)) {
      return { passed: false, output: `expected code ${String(args.code)}, got ${err.code}` }
    }
    if (
      args.messageContains !== undefined &&
      !message.toLowerCase().includes(String(args.messageContains).toLowerCase())
    ) {
      return {
        passed: false,
        output: `message does not contain "${String(args.messageContains)}": ${message}`
      }
    }
    if (
      args.messageNotMatching !== undefined &&
      new RegExp(String(args.messageNotMatching), 'i').test(message)
    ) {
      return { passed: false, output: `message matched the forbidden pattern: ${message}` }
    }
    return { passed: true, output: `code=${err.code || '(none)'}: ${message.slice(0, 120)}` }
  },

  /**
   * The model made a structured tool call, and the right one.
   *
   * `declared` is the tools the test offered: a call naming something that was
   * never declared is a failure however well-formed it looks, and that check
   * is the reason this is not an ordinary field comparison.
   */
  toolCallShape(value, args) {
    const calls = (Array.isArray(value) ? value : []) as Array<{
      name?: string
      arguments?: Record<string, unknown>
    }>
    if (calls.length === 0) {
      return { passed: false, output: 'expected a structured tool call but the model made none' }
    }

    const declared = new Set((args.declared ?? []) as string[])
    const valid = calls.filter((call) => typeof call.name === 'string' && declared.has(call.name))
    if (valid.length === 0) {
      return {
        passed: false,
        output:
          `no tool call matched a declared tool. Got: [${calls.map((c) => c.name ?? '<unnamed>').join(', ')}], ` +
          `declared: [${[...declared].join(', ')}]`
      }
    }

    const match = valid.find((call) => call.name === args.name)
    if (match) {
      const callArgs = match.arguments ?? {}
      for (const key of (args.argKeys ?? []) as string[]) {
        if (!(key in callArgs)) {
          return {
            passed: false,
            output: `tool call '${String(args.name)}' is missing argument '${key}': ${JSON.stringify(callArgs)}`
          }
        }
      }
    }

    return { passed: true, output: `tool call(s): ${valid.map((call) => call.name).join(', ')}` }
  },

  /**
   * Every block carries the geometry a caller needs to place it.
   *
   * The OCR executors checked this inline; as a named assertion it is the same
   * check on every client, which is the difference between two clients
   * agreeing and two clients each having an opinion.
   */
  textBlockShape(value) {
    const blocks = (Array.isArray(value) ? value : []) as Array<{
      text?: unknown
      bbox?: unknown
      confidence?: unknown
    }>
    for (const [index, block] of blocks.entries()) {
      if (typeof block.text !== 'string') {
        return { passed: false, output: `block[${index}].text is not a string` }
      }
      const bbox = block.bbox
      if (!Array.isArray(bbox) || bbox.length !== 4) {
        return { passed: false, output: `block[${index}].bbox is not a 4-element array` }
      }
      const bad = bbox.findIndex((coordinate) => typeof coordinate !== 'number')
      if (bad !== -1) {
        return { passed: false, output: `block[${index}].bbox[${bad}] is not a number` }
      }
      if (typeof block.confidence !== 'number') {
        return { passed: false, output: `block[${index}].confidence is not a number` }
      }
    }
    return { passed: true, output: `${blocks.length} well-formed block(s)` }
  },

  /**
   * The run reported how long it took.
   *
   * `field` names which timing to insist on, because the engines do not agree
   * on what they measure -- and a test that only checks "stats exist" passes
   * on an object full of nulls.
   */
  timingStatsPresent(value, args) {
    if (!value || typeof value !== 'object') {
      return { passed: false, output: 'stats is undefined, expected timing data' }
    }
    const stats = value as Record<string, unknown>
    const field = String(args.field ?? 'totalTime')
    const measured = stats[field]
    if (typeof measured !== 'number' || measured <= 0) {
      return {
        passed: false,
        output: `expected stats.${field} > 0, got ${JSON.stringify(measured)}`
      }
    }
    return { passed: true, output: `${field}=${measured}` }
  },

  /**
   * The run produced audio.
   *
   * `minSamples` is the bar: 1 for a normal synthesis, 0 for the tests that
   * feed empty text and only care that the SDK handled it rather than
   * crashing. The executors asserted a synthesised sentence -- "generated N
   * samples" -- against `type: string`, which every string satisfies, so they
   * could not fail whatever the engine did. This asks the question they meant.
   */
  producedAudio(value, args) {
    const samples = Array.isArray(value)
      ? value.length
      : ArrayBuffer.isView(value)
        ? (value as unknown as { length: number }).length
        : 0
    const floor = Number(args.minSamples ?? 1)
    if (samples < floor) {
      return { passed: false, output: `expected at least ${floor} sample(s), got ${samples}` }
    }
    return { passed: true, output: `${samples} sample(s)` }
  },

  /**
   * The value is exactly `true`.
   *
   * For the operations whose whole answer is "it worked": the executors turned
   * that into the string "success" and matched it against `type: string`,
   * which is satisfied by "failed" just as well.
   */
  isTrue(value) {
    if (value !== true) {
      return { passed: false, output: `expected true, got ${JSON.stringify(value)}` }
    }
    return { passed: true, output: 'true' }
  },

  /**
   * Every named field is a positive integer.
   *
   * Model hyper-parameters are the recurring case: a chunk size or an action
   * dimension that arrives as 0, a float, or a string is a broken model
   * description however well-formed the surrounding object looks.
   */
  positiveIntegers(value, args) {
    const record = (value ?? {}) as Record<string, unknown>
    for (const field of (args.fields ?? []) as string[]) {
      const measured = record[field]
      if (typeof measured !== 'number' || !Number.isInteger(measured) || measured <= 0) {
        return {
          passed: false,
          output: `${field} is not a positive integer (got ${JSON.stringify(measured)})`
        }
      }
    }
    return { passed: true, output: `${(args.fields as string[]).length} field(s) positive` }
  },

  /** The value is one of a known set. `allowNull` admits "not reported". */
  valueIn(value, args) {
    if (value === null && args.allowNull) return { passed: true, output: 'null' }
    const allowed = (args.values ?? []) as unknown[]
    if (!allowed.includes(value)) {
      return {
        passed: false,
        output: `${JSON.stringify(value)} is not one of ${JSON.stringify(allowed)}`
      }
    }
    return { passed: true, output: String(value) }
  },

  /**
   * Two fields of the same object agree.
   *
   * For the invariants a result carries about itself -- a buffer whose length
   * must equal the product of the dimensions reported beside it.
   */
  fieldEquals(value, args) {
    const record = (value ?? {}) as Record<string, unknown>
    const left = record[String(args.field)]
    const right = record[String(args.other)]
    if (left !== right) {
      return {
        passed: false,
        output: `${String(args.field)}=${JSON.stringify(left)} != ${String(args.other)}=${JSON.stringify(right)}`
      }
    }
    return { passed: true, output: `${String(args.field)} == ${String(args.other)}` }
  },

  loadedModelInfoShape(value, args) {
    const info = value as {
      modelId?: string
      modelType?: string
      handlers?: string[]
    }

    const checks: Record<string, boolean> = {
      modelIdMatches: info.modelId === args.expectedModelId,
      handlersIsList: Array.isArray(info.handlers),
      modelTypePresent: typeof info.modelType === 'string' && info.modelType.length > 0
    }
    if (args.handlerIncludes !== undefined) {
      checks.handlerPresent = (info.handlers ?? []).includes(args.handlerIncludes as string)
    }

    const failed = Object.entries(checks)
      .filter(([, ok]) => !ok)
      .map(([name]) => name)

    if (failed.length > 0) {
      return {
        passed: false,
        output:
          `loadedModelInfoShape failed: ${failed.join(', ')} ` +
          `(modelId=${info.modelId}, modelType=${info.modelType}, handlers=${JSON.stringify(info.handlers)})`
      }
    }

    return {
      passed: true,
      output: `modelType=${info.modelType}, handlers=${info.handlers?.length ?? 0}`
    }
  }
}

export function createStepBindings(resources: ResourceManager): StepBindings {
  return {
    async useModel(deps) {
      const ids: string[] = []
      for (const dep of deps) {
        ids.push(await resources.ensureLoaded(dep))
      }
      return ids
    },

    async call(method, params, collect) {
      if (collect) {
        const stream = STREAMS[method]
        if (!stream) {
          // A gap in the bindings, not a failing test, so say so explicitly —
          // a plain Error would be reported as a failure.
          throw new StepIncompleteError(
            `SDK method "${method}" has no stream fold in these bindings yet`
          )
        }
        return stream(params as never, collect)
      }

      const call = CALLS[method]
      if (!call) {
        throw new StepIncompleteError(`SDK method "${method}" is not wired into these bindings yet`)
      }
      return call(params as never)
    },

    async modelSource(dep) {
      return resources.sourceOf(dep)
    },

    async asset(kind, file, form) {
      if (kind === 'bytes') {
        if (form === 'path') throw new Error('synthesized bytes have no path')
        return synthesizeBytes(file)
      }
      if (kind === 'tone') {
        if (form === 'path') throw new Error('a synthesized tone has no path')
        return synthesizeTone(file)
      }
      const root = ASSET_ROOTS[kind]
      if (!root) {
        throw new StepIncompleteError(`asset kind "${kind}" is not known to these bindings`)
      }
      const base = path.resolve(process.cwd(), root)
      const absolute = path.resolve(base, file)
      // The asset name arrives from the catalog, and a definition is data that
      // travels between clients. A name that climbs out of the asset root must
      // be refused here rather than trusted because today's catalog happens to
      // contain only literals.
      if (absolute !== base && !absolute.startsWith(base + path.sep)) {
        throw new Error(`asset "${file}" resolves outside the "${kind}" asset root`)
      }
      if (!fs.existsSync(absolute)) {
        // A missing fixture is a real failure, not a client gap.
        throw new Error(`asset not found: ${absolute}`)
      }
      // On desktop the path form is a filesystem path; a mobile binding hands
      // back a bundled-asset URI for the same pair, which is the entire reason
      // several categories still carry two executors.
      if (form === 'path') return absolute
      if (form === 'text') return fs.readFileSync(absolute, 'utf-8')
      return new Uint8Array(fs.readFileSync(absolute))
    },

    assertions: ASSERTIONS,

    comparisons: COMPARISONS,

    async evictAllExcept(keep) {
      await resources.evictExcept([...keep])
    }
  }
}
