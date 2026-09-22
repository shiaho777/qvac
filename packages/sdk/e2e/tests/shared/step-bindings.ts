import * as fs from 'node:fs'
import * as path from 'node:path'
import {
  classify,
  completion,
  deleteCache,
  embed,
  getLoadedModelInfo,
  loadModel,
  modelRegistryGetModel,
  modelRegistryList,
  modelRegistrySearch,
  ragIngest,
  transcribe
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
  embed: (params) => embed(params),
  getLoadedModelInfo: (params) => getLoadedModelInfo(params),
  classify: async (params) => ({ results: await classify(params) }),

  // The registry trio takes its arguments differently in each language --
  // positional here, keyword in Python. The contract name and the params
  // object in the step are what both sides agree on; adapting to the local
  // signature is precisely what a binding is for.
  modelRegistryList: () => modelRegistryList(),
  modelRegistrySearch: (params) => modelRegistrySearch(params),
  modelRegistryGetModel: (params: never) => {
    const p = params as unknown as { registryPath: string; registrySource: string }
    return modelRegistryGetModel(p.registryPath, p.registrySource)
  },

  loadModel: (params) => loadModel(params),
  deleteCache: (params) => deleteCache(params),
  ragIngest: (params) => ragIngest(params),
  transcribe: (params) => transcribe(params)
}

/**
 * Methods whose result is a stream handle rather than a value.
 *
 * A step reaches these through `collect`, which names the fold it wants. The
 * fold is the interesting part of the binding: two clients are only running
 * the same test if "the text of this completion" means the same thing on both.
 */
const STREAMS: Record<string, (params: never, collect: CollectMode) => Promise<unknown>> = {
  completion: async (params, collect) => {
    const run = completion(params)
    if (collect === 'text') return { text: await run.text }
    if (collect === 'events') {
      const events: unknown[] = []
      for await (const event of run.events) events.push(event)
      return { events }
    }
    throw new StepIncompleteError(`collect: "${collect}" is not defined for completion`)
  }
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

    async asset(kind, file) {
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
      return new Uint8Array(fs.readFileSync(absolute))
    },

    assertions: ASSERTIONS,

    async evictAllExcept(keep) {
      await resources.evictExcept([...keep])
    }
  }
}
