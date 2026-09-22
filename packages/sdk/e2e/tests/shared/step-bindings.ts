import * as fs from 'node:fs'
import * as path from 'node:path'
import { classify, embed, getLoadedModelInfo } from '@qvac/sdk'
import { StepIncompleteError, type StepBindings } from '@qvac/test-suite'
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
  classify: async (params) => ({ results: await classify(params) })
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
        // Stream folds are not part of the first vocabulary slice. This is a
        // gap in the bindings, not a failing test, so say so explicitly — a
        // plain Error would be reported as a failure.
        throw new StepIncompleteError(
          `collect: "${collect}" is not implemented by these bindings yet`
        )
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
