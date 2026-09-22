import type { Step, TestDefinition } from '@qvac/test-suite'

/**
 * A call that must reject, with its message checked against the expectation.
 *
 * Almost every test in this category is this shape. Writing it once is the
 * point of the vocabulary: the migrated definitions differ only in which call
 * they expect to fail, which is the only thing they ever differed in.
 */
const rejects = (
  method: string,
  params: Record<string, unknown>,
  options: { collect?: 'text' | 'events'; before?: Step[] } = {}
): Step[] => [
  ...(options.before ?? []),
  {
    callError: {
      method,
      params,
      ...(options.collect ? { collect: options.collect } : {}),
      as: 'err'
    }
  },
  { project: { from: '$err', path: 'message', as: 'message' } },
  { assert: { on: '$message', use: 'expectation' } }
]

/**
 * The four generation-parameter tests below are NOT migrated, and that is a
 * finding rather than an omission.
 *
 * Each declares `throws-error` with `errorContains: ''`, and the executor
 * validates the completion's *text* against that expectation when the call
 * succeeds. Every string contains the empty string, so all four passed whether
 * the SDK rejected the parameter or happily generated from it. Writing them as
 * `callError` made them fail immediately: the SDK forwards `generationParams`
 * to the worker without range checks, so `temp: -0.5`, `temp: 3.0`,
 * `top_p: 1.5` and `predict: -10` all complete normally.
 *
 * Whether the SDK should reject those is a product decision, not one to settle
 * inside a catalog migration, so their behaviour is left exactly as it was
 * until it is. Migrating them is what surfaced it -- which is the argument for
 * migrating the rest.
 */

export const errorInvalidModelId: TestDefinition = {
  testId: 'error-invalid-model-id',
  params: { modelId: 'nonexistent-model-id-12345', operation: 'embed', text: 'test text' },
  expectation: { validation: 'throws-error', errorContains: '' },
  suites: ['smoke'],
  steps: rejects('embed', { modelId: '$params.modelId', text: '$params.text' }),
  metadata: { category: 'error', dependency: 'embeddings', estimatedDurationMs: 5000 }
}

/**
 * Reads the SDK's exported error-code tables rather than calling anything.
 *
 * Left on the executor deliberately: there is no call to make, and the tables
 * are a property of one client's module surface, not of the shared contract.
 * A step vocabulary that could express "read this module constant" would be
 * expressing the JS package, which is the opposite of what it is for.
 */
export const errorInvalidResponseType: TestDefinition = {
  testId: 'error-invalid-response-type',
  params: { verifyErrorCodes: true },
  expectation: { validation: 'type', expectedType: 'string' },
  metadata: { category: 'error', dependency: 'none', estimatedDurationMs: 2000 }
}

export const errorModelLoadFailed: TestDefinition = {
  testId: 'error-model-load-failed',
  params: { modelPath: '/invalid/path/to/model.gguf', modelType: 'llamacpp-completion' },
  expectation: { validation: 'throws-error', errorContains: '' },
  steps: rejects('loadModel', {
    modelSrc: '$params.modelPath',
    modelType: '$params.modelType'
  }),
  metadata: { category: 'error', dependency: 'none', estimatedDurationMs: 5000 }
}

export const errorDeleteCacheInvalidParams: TestDefinition = {
  testId: 'error-delete-cache-invalid-params',
  params: { invalidParams: true },
  expectation: { validation: 'throws-error', errorContains: '' },
  steps: rejects('deleteCache', {}),
  metadata: { category: 'error', dependency: 'none', estimatedDurationMs: 5000 }
}

/** See `errorInvalidResponseType`: a module-surface test, not a contract test. */
export const errorStructuredErrorCode: TestDefinition = {
  testId: 'error-structured-error-code',
  params: { verifyErrorCodes: true },
  expectation: { validation: 'type', expectedType: 'string' },
  suites: ['smoke'],
  metadata: { category: 'error', dependency: 'none', estimatedDurationMs: 2000 }
}

export const errorChainingCause: TestDefinition = {
  testId: 'error-chaining-cause',
  params: {
    triggerChainedError: true,
    modelPath: '/invalid/nonexistent/path/model.gguf',
    modelType: 'llamacpp-completion'
  },
  expectation: { validation: 'throws-error', errorContains: '' },
  // The question is whether the rejection carries structure, so the named
  // assertion reads the binding rather than the message.
  steps: [
    {
      callError: {
        method: 'loadModel',
        params: { modelSrc: '$params.modelPath', modelType: '$params.modelType' },
        as: 'err'
      }
    },
    { assert: { on: '$err', named: 'errorIsStructured' } }
  ],
  metadata: { category: 'error', dependency: 'none', estimatedDurationMs: 5000 }
}

export const errorRagOperationFailed: TestDefinition = {
  testId: 'error-rag-operation-failed',
  params: {
    modelId: 'nonexistent-model',
    query: 'test query',
    documents: 'test content',
    workspace: 'test'
  },
  expectation: { validation: 'throws-error', errorContains: '' },
  suites: ['smoke'],
  steps: rejects('ragIngest', {
    modelId: '$params.modelId',
    documents: '$params.documents',
    workspace: '$params.workspace'
  }),
  metadata: { category: 'error', dependency: 'embeddings', estimatedDurationMs: 5000 }
}

export const errorTranscriptionFailed: TestDefinition = {
  testId: 'error-transcription-failed',
  params: { audioPath: '/nonexistent/audio/file.wav' },
  expectation: { validation: 'throws-error', errorContains: '' },
  steps: rejects(
    'transcribe',
    { modelId: '$model', audioChunk: '$params.audioPath' },
    { before: [{ useModel: { deps: ['whisper'], as: 'model' } }] }
  ),
  metadata: { category: 'error', dependency: 'whisper', estimatedDurationMs: 5000 }
}

export const errorCompletionNegativeTemperature: TestDefinition = {
  testId: 'error-completion-negative-temperature',
  params: {
    history: [{ role: 'user', content: 'Test' }],
    stream: false,
    generationParams: { temp: -0.5 }
  },
  expectation: { validation: 'throws-error', errorContains: '' },
  suites: ['smoke'],
  metadata: { category: 'error', dependency: 'llm', estimatedDurationMs: 3000 }
}

export const errorCompletionExcessiveTemperature: TestDefinition = {
  testId: 'error-completion-excessive-temperature',
  params: {
    history: [{ role: 'user', content: 'Test' }],
    stream: false,
    generationParams: { temp: 3.0 }
  },
  expectation: { validation: 'throws-error', errorContains: '' },
  metadata: { category: 'error', dependency: 'llm', estimatedDurationMs: 3000 }
}

export const errorCompletionInvalidTopP: TestDefinition = {
  testId: 'error-completion-invalid-topp',
  params: {
    history: [{ role: 'user', content: 'Test' }],
    stream: false,
    generationParams: { top_p: 1.5 }
  },
  expectation: { validation: 'throws-error', errorContains: '' },
  metadata: { category: 'error', dependency: 'llm', estimatedDurationMs: 3000 }
}

export const errorCompletionNegativeMaxTokens: TestDefinition = {
  testId: 'error-completion-negative-maxtokens',
  params: {
    history: [{ role: 'user', content: 'Test' }],
    stream: false,
    generationParams: { predict: -10 }
  },
  expectation: { validation: 'throws-error', errorContains: '' },
  metadata: { category: 'error', dependency: 'llm', estimatedDurationMs: 3000 }
}

export const errorEmbeddingEmptyInput: TestDefinition = {
  testId: 'error-embedding-empty-input',
  params: { text: ' ' },
  expectation: { validation: 'type', expectedType: 'array' },
  // The executor passed whether the SDK embedded whitespace or rejected it,
  // so it could not fail. The observed behaviour is that it embeds -- the same
  // thing `embed-empty-text` asserts -- so the migrated body asserts that, and
  // a change of behaviour now shows up instead of passing silently.
  steps: [
    { useModel: { deps: ['embeddings'], as: 'model' } },
    {
      call: { method: 'embed', params: { modelId: '$model', text: '$params.text' }, as: 'response' }
    },
    { project: { from: '$response', path: 'embedding', as: 'embedding' } },
    { assert: { on: '$embedding', use: 'expectation' } }
  ],
  metadata: { category: 'error', dependency: 'embeddings', estimatedDurationMs: 3000 }
}

export const errorUseUnloadedModel: TestDefinition = {
  testId: 'error-use-unloaded-model',
  params: {
    modelIdOverride: 'unloaded-model-id-12345',
    history: [{ role: 'user', content: 'Test' }],
    stream: false
  },
  expectation: { validation: 'throws-error', errorContains: '' },
  steps: rejects(
    'completion',
    {
      modelId: '$params.modelIdOverride',
      history: '$params.history',
      stream: '$params.stream'
    },
    { collect: 'text' }
  ),
  suites: ['smoke'],
  metadata: { category: 'error', dependency: 'llm', estimatedDurationMs: 3000 }
}

export const errorRagUnloadedModel: TestDefinition = {
  testId: 'error-rag-unloaded-model',
  params: {
    modelIdOverride: 'unloaded-embedding-model-xyz',
    documentFile: 'ocean_waves_poem.txt',
    chunkSize: 200,
    chunkOverlap: 50,
    documents: 'test',
    workspace: 'test'
  },
  expectation: { validation: 'throws-error', errorContains: '' },
  steps: rejects('ragIngest', {
    modelId: '$params.modelIdOverride',
    documents: '$params.documents',
    workspace: '$params.workspace'
  }),
  metadata: { category: 'error', dependency: 'embeddings', estimatedDurationMs: 3000 }
}

export const errorTests = [
  errorInvalidModelId,
  errorInvalidResponseType,
  errorModelLoadFailed,
  errorDeleteCacheInvalidParams,
  errorStructuredErrorCode,
  errorChainingCause,
  errorRagOperationFailed,
  errorTranscriptionFailed,
  errorCompletionNegativeTemperature,
  errorCompletionExcessiveTemperature,
  errorCompletionInvalidTopP,
  errorCompletionNegativeMaxTokens,
  errorEmbeddingEmptyInput,
  errorUseUnloadedModel,
  errorRagUnloadedModel
]
