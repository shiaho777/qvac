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
  worldStep
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
    if (collect === 'all') return { all: await drain(run.tokenStream) }
    if (collect !== 'text') throw unsupported('translate', collect)
    // `text` resolves to the empty string in streaming mode on both clients, so
    // the fold has to follow the mode rather than always await the same handle.
    if (!p.stream) return { text: await run.text }
    return { text: await joinStream(run.tokenStream) }
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
    if (collect === 'all') return { all: await run.results }
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
  if (collect === 'pcm') return { pcm: await run.audio, stats: await run.stats }
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
const ASSET_ROOTS: Record<string, string> = {
  image: 'assets/images',
  audio: 'assets/audio',
  document: 'assets/documents',
  neural: 'assets/neural'
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
      return form === 'path' ? absolute : new Uint8Array(fs.readFileSync(absolute))
    },

    assertions: ASSERTIONS,

    async evictAllExcept(keep) {
      await resources.evictExcept([...keep])
    }
  }
}
