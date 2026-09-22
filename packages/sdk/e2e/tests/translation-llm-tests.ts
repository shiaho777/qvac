import type { Step, TestDefinition } from '@qvac/test-suite'

/**
 * One LLM-backed translation, folded to its text.
 *
 * `from` and `context` are optional references: left out when the test does
 * not set them, which is how the autodetect case says "no source language"
 * without needing a body of its own. `stream: false` because `text` is the
 * handle that carries the result in non-streaming mode on both clients -- the
 * streaming case reads the token stream instead and stays on the executor
 * until the vocabulary can say "and it arrived in more than one piece".
 */
const translateSteps = (extra: Step[] = []): Step[] => [
  { useModel: { deps: ['llm'], as: 'model' } },
  {
    call: {
      method: 'translate',
      collect: 'text',
      params: {
        modelId: '$model',
        text: '$params.text',
        to: '$params.to',
        from: '$params.from?',
        context: '$params.context?',
        modelType: 'llamacpp-completion',
        stream: false
      },
      as: 'run'
    }
  },
  { project: { from: '$run', path: 'text', as: 'text' } },
  { assert: { on: '$text', use: 'expectation' } },
  ...extra
]

/** The executor additionally required real output from these two. */
const producesText: Step[] = [{ assert: { on: '$text', named: 'nonEmptyText' } }]

const createLlmTest = (
  testId: string,
  text: string,
  to: string,
  opts: { from?: string; context?: string; estimatedDurationMs?: number } = {},
  suites?: string[]
): TestDefinition => ({
  testId,
  params: {
    text,
    to,
    resource: 'llm',
    ...(opts.from && { from: opts.from }),
    ...(opts.context && { context: opts.context })
  },
  expectation: { validation: 'type', expectedType: 'string' },
  ...(suites && { suites }),
  steps: translateSteps(opts.context ? producesText : []),
  metadata: {
    category: 'translation-llm',
    dependency: 'llm',
    estimatedDurationMs: opts.estimatedDurationMs ?? 90000
  }
})

export const llmEnEs = createLlmTest(
  'translation-llm-en-es',
  'Hello, how are you today?',
  'es',
  { from: 'en' },
  ['smoke']
)

export const llmEnFr = createLlmTest('translation-llm-en-fr', 'Good morning, how are you?', 'fr', {
  from: 'en'
})

export const llmEsEn = createLlmTest(
  'translation-llm-es-en',
  'Buenos días, ¿cómo estás hoy?',
  'en',
  { from: 'es' }
)

export const llmAutodetect: TestDefinition = {
  testId: 'translation-llm-autodetect',
  params: { text: "Bonjour, comment allez-vous aujourd'hui?", to: 'en', resource: 'llm' },
  expectation: { validation: 'type', expectedType: 'string' },
  // No `from`: the optional reference resolves to nothing and the argument is
  // left off the call, so the worker detects the source language.
  steps: translateSteps(producesText),
  metadata: { category: 'translation-llm', dependency: 'llm', estimatedDurationMs: 90000 }
}

export const llmStreaming: TestDefinition = {
  testId: 'translation-llm-streaming',
  params: { text: 'Hello, how are you today?', from: 'en', to: 'es', resource: 'llm' },
  expectation: { validation: 'type', expectedType: 'string' },
  suites: ['smoke'],
  metadata: { category: 'translation-llm', dependency: 'llm', estimatedDurationMs: 30000 }
}

export const llmStats: TestDefinition = {
  testId: 'translation-llm-stats',
  params: { text: 'Hello world', from: 'en', to: 'es', resource: 'llm' },
  expectation: { validation: 'type', expectedType: 'string' },
  metadata: { category: 'translation-llm', dependency: 'llm', estimatedDurationMs: 30000 }
}

export const llmContext = createLlmTest('translation-llm-context', 'bank', 'es', {
  from: 'en',
  context: 'Use formal language, context is financial institution'
})

export const llmLongText = createLlmTest(
  'translation-llm-long-text',
  'The weather is beautiful today. I decided to go for a walk in the park. The birds are singing and the flowers are blooming. It is a perfect day to enjoy nature and relax.',
  'es',
  { from: 'en', estimatedDurationMs: 45000 }
)

export const llmEmptyText: TestDefinition = {
  testId: 'translation-llm-empty-text',
  params: { text: '', from: 'en', to: 'es', resource: 'llm' },
  expectation: { validation: 'type', expectedType: 'string' },
  metadata: { category: 'translation-llm', dependency: 'llm', estimatedDurationMs: 15000 }
}

export const translationLlmTests = [
  llmEnEs,
  llmEnFr,
  llmEsEn,
  llmAutodetect,
  llmStreaming,
  llmStats,
  llmContext,
  llmLongText,
  llmEmptyText
]
