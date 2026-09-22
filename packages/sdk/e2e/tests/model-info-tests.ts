import type { TestDefinition } from '@qvac/test-suite'

export const modelInfoGet: TestDefinition = {
  testId: 'model-info-get',
  params: { modelConstant: 'LLAMA_3_2_1B_INST_Q4_0' },
  expectation: { validation: 'type', expectedType: 'string' },
  suites: ['smoke'],
  metadata: { category: 'model-info', dependency: 'llm', estimatedDurationMs: 5000 }
}

export const modelInfoVerifyFiles: TestDefinition = {
  testId: 'model-info-verify-files',
  params: { modelConstant: 'LLAMA_3_2_1B_INST_Q4_0' },
  expectation: { validation: 'type', expectedType: 'string' },
  metadata: { category: 'model-info', dependency: 'llm', estimatedDurationMs: 5000 }
}

export const modelInfoMultipleModels: TestDefinition = {
  testId: 'model-info-multiple-models',
  params: { models: ['LLAMA_3_2_1B_INST_Q4_0', 'GTE_LARGE_FP16'] },
  expectation: { validation: 'type', expectedType: 'string' },
  metadata: { category: 'model-info', dependency: 'llm+embeddings', estimatedDurationMs: 10000 }
}

export const modelInfoPersistsAfterUnload: TestDefinition = {
  testId: 'model-info-persists-after-unload',
  params: { modelConstant: 'LLAMA_3_2_1B_INST_Q4_0' },
  expectation: { validation: 'type', expectedType: 'string' },
  suites: ['smoke'],
  metadata: { category: 'model-info', dependency: 'llm', estimatedDurationMs: 5000 }
}

export const modelInfoLoadedGet: TestDefinition = {
  testId: 'model-info-loaded-get',
  params: {},
  expectation: { validation: 'type', expectedType: 'string' },
  suites: ['smoke'],
  // The executor checked the returned record inline — that the modelId is the
  // one we just loaded, that the type is canonical, that the handlers include
  // completionStream. As data, those become one named assertion, and the id to
  // compare against is handed over with `with`.
  steps: [
    { useModel: { deps: ['llm'], as: 'model' } },
    { call: { method: 'getLoadedModelInfo', params: { modelId: '$model' }, as: 'info' } },
    {
      assert: {
        on: '$info',
        named: 'loadedModelInfoShape',
        with: { expectedModelId: '$model', handlerIncludes: 'completionStream' }
      }
    }
  ],
  metadata: { category: 'model-info', dependency: 'llm', estimatedDurationMs: 5000 }
}

export const modelInfoLoadedNotFound: TestDefinition = {
  testId: 'model-info-loaded-not-found',
  params: { modelId: 'nonexistent-model-id-deadbeef' },
  expectation: { validation: 'throws-error', errorContains: 'not found' },
  suites: ['smoke'],
  // A call expected to fail. If it ever stops failing the test fails, which is
  // the point: an error test that quietly passes when the error stops
  // happening is worse than no test.
  steps: [
    {
      callError: {
        method: 'getLoadedModelInfo',
        params: { modelId: '$params.modelId' },
        as: 'err'
      }
    },
    { project: { from: '$err', path: 'message', as: 'message' } },
    { assert: { on: '$message', use: 'expectation' } }
  ],
  metadata: { category: 'model-info', dependency: 'none', estimatedDurationMs: 2000 }
}

export const modelInfoTests = [
  modelInfoGet,
  modelInfoVerifyFiles,
  modelInfoMultipleModels,
  modelInfoPersistsAfterUnload,
  modelInfoLoadedGet,
  modelInfoLoadedNotFound
]
