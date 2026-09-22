import { createExecutor, SkipExecutor, type TestDefinition } from '@qvac/test-suite'
import { createStepBindings } from '../shared/step-bindings.js'
import {
  profiler,
  LLAMA_3_2_1B_INST_Q4_0,
  LLAMA_3_2_1B_INST_Q4_0_SHARD,
  GTE_LARGE_FP16,
  GTE_LARGE_335M_FP16_SHARD,
  WHISPER_TINY,
  VAD_SILERO_5_1_2,
  QWEN3_1_7B_INST_Q4,
  OCR_CRAFT,
  OCR_LATIN,
  OCR_DOCTR,
  BERGAMOT_EN_FR,
  BERGAMOT_EN_ES,
  BERGAMOT_ES_EN,
  BERGAMOT_EN_IT,
  MARIAN_EN_HI_INDIC_200M_Q4_0,
  MARIAN_HI_EN_INDIC_200M_Q4_0,
  TTS_T3_TURBO_EN_CHATTERBOX_Q4_0,
  TTS_S3GEN_EN_CHATTERBOX_Q4_0,
  TTS_INDIC_MULTILINGUAL_PARLER_TTS_Q8_0,
  TTS_MINI_V1_EN_PARLER_TTS_Q8_0,
  TTS_COSYVOICE3_LLM_COSYVOICE_Q8_0,
  TTS_LM_MULTILINGUAL_AUDIO8_Q8_0,
  TTS_CODEC_DECODER_AUDIO8_Q8_0,
  TTS_EN_SUPERTONIC_Q8_0,
  TTS_MULTILINGUAL_SUPERTONIC3_Q4_0,
  TTS_ENHANCER_LAVASR_FP16,
  TTS_DENOISER_LAVASR_FP16,
  PARAKEET_TDT_0_6B_V3_Q4_0,
  PARAKEET_CTC_0_6B_Q4_0,
  PARAKEET_UNIFIED_0_6B_Q4_0,
  PARAKEET_INDIC_CONFORMER_CTC_Q4_0,
  PARAKEET_SORTFORMER_4SPK_V2_1_Q4_0,
  PARAKEET_EOU_120M_V1_Q4_0,
  SMOLVLA_LIBERO_VISION_Q8,
  PI05_BASE_Q_AGGRESSIVE,
  GROOT_Q5_VF16,
  GROOT_MULTI_Q5_VF16,
  VISIONPSY_NANO_460M_MULTIMODAL_Q4_K_M,
  MMPROJ_VISIONPSY_NANO_460M_MULTIMODAL_Q8_0,
  FLUX_2_KLEIN_4B_Q4_0,
  ABOT_WORLD_0_5B_Q8_0,
  ABOT_WORLD_0_5B_LF_TAEHV_VAE,
  ABOT_WORLD_0_5B_LF_WAN_VAE,
  UMT5_XXL_ENC_Q8_0,
  FLUX_2_KLEIN_4B_VAE,
  QWEN3_4B_Q4_K_M,
  SD_V2_1_1B_Q8_0,
  REALESRGAN_X4PLUS_ANIME_6B,
  QWEN3_5_0_8B_MULTIMODAL_Q4_K_M,
  QWEN3_5_0_8B_MULTIMODAL_Q8_0,
  GEMMA4_2B_MULTIMODAL_Q4_K_M,
  BCI_WINDOWED,
  AUDIOGEN_QWEN3_EMBEDDING_0_6B_Q8_0,
  AUDIOGEN_ACESTEP_5HZ_LM_0_6B_Q8_0,
  AUDIOGEN_ACESTEP_V15_TURBO_Q4_K_M,
  AUDIOGEN_VAE_BF16
} from '@qvac/sdk'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { ResourceManager } from '../shared/resource-manager.js'
import { collectTestDeps } from '../shared/collect-test-deps.js'
import { BatchCompletionExecutor } from '../shared/executors/batch-completion-executor.js'
import { ModelLoadingExecutor } from '../shared/executors/model-loading-executor.js'
import { CompletionExecutor } from '../shared/executors/completion-executor.js'
import { ToolsExecutor } from '../shared/executors/tools-executor.js'
import { TranslationExecutor } from '../shared/executors/translation-executor.js'
import { TranslationBergamotCacheExecutor } from '../shared/executors/translation-bergamot-cache-executor.js'
import { ShardedModelExecutor } from '../shared/executors/sharded-model-executor.js'
import { HttpEmbeddingExecutor } from '../shared/executors/http-embedding-executor.js'
import { KvCacheExecutor } from '../shared/executors/kv-cache-executor.js'
import { EmbeddingExecutor } from '../shared/executors/embedding-executor.js'
import { TranscriptionExecutor } from '../shared/executors/node/transcription-executor.js'
import { TranscribeStreamEventsExecutor } from '../shared/executors/node/transcribe-stream-events-executor.js'
import { RagExecutor } from '../shared/executors/node/rag-executor.js'
import { VectorIndexExecutor } from '../shared/executors/vector-index-executor.js'
import { OcrExecutor } from '../shared/executors/node/ocr-executor.js'
import { VlaExecutor } from '../shared/executors/vla-executor.js'
import { ClassificationExecutor } from '../shared/executors/node/classification-executor.js'
import { ConfigReloadExecutor } from '../shared/executors/node/config-reload-executor.js'
import { NodeLoggingExecutor } from '../shared/executors/node/logging-executor.js'
import { RegistryExecutor } from '../shared/executors/registry-executor.js'
import { ModelInfoExecutor } from '../shared/executors/model-info-executor.js'
import { WrongModelExecutor } from '../shared/executors/wrong-model-executor.js'
import { ErrorExecutor } from '../shared/executors/error-executor.js'
import { TtsExecutor } from '../shared/executors/tts-executor.js'
import { ParakeetStreamExecutor } from '../shared/executors/node/parakeet-stream-executor.js'
import { ParakeetExecutor } from '../shared/executors/node/parakeet-executor.js'
import { BciExecutor } from '../shared/executors/node/bci-executor.js'
import { VisionExecutor } from '../shared/executors/node/vision-executor.js'
import { DownloadExecutor } from '../shared/executors/download-executor.js'
import { DownloadResilienceExecutor } from '../shared/executors/node/download-resilience-executor.js'
import { NodeDiffusionExecutor } from '../shared/executors/node/diffusion-executor.js'
import { NodeWorldExecutor } from '../shared/executors/node/world-executor.js'
import { AudioGenExecutor } from '../shared/executors/audio-gen-executor.js'
import { FinetuneExecutor } from '../shared/executors/node/finetune-executor.js'
import { LifecycleExecutor } from '../shared/executors/lifecycle-executor.js'
import { SystemResourcesExecutor } from '../shared/executors/system-resources-executor.js'
import { ConfigExecutor } from '../shared/executors/config-executor.js'
import { NoLingeringBareExecutor } from '../shared/executors/node/no-lingering-bare-executor.js'
import { KvCacheRestartExecutor } from '../shared/executors/node/kv-cache-restart-executor.js'
import { MultiGpuExecutor } from '../shared/executors/multi-gpu-executor.js'
import { NodeCancellationExecutor } from '../shared/executors/node/cancellation-executor.js'
import { PluginExecutor } from '../shared/executors/plugin-executor.js'

import * as MODEL_CONSTANTS from '@qvac/sdk'
import { RESOURCE_TABLE } from '../shared/resource-table.js'
import { applyResourceTable } from '../shared/resource-table-types.js'

/** Where the shared table's `$asset` placeholders point on this platform. */
function resolveTableAsset(kind: string, file: string): string {
  return path.resolve(process.cwd(), `assets/${kind}`, file)
}

const resources = new ResourceManager({
  downloadTarget: 'desktop'
})

// One table, shared with every other client, applied here.
//
// Which model `llm` or `tts-supertonic` means used to be written out once per
// consumer entry and nowhere a client in another language could read it. A
// definition that says `useModel: { deps: ['whisper'] }` only means the same
// thing on two clients if both resolve the key the same way, so the table is
// data now and this is the part that cannot be: the descriptor behind a model
// constant, and where a bundled fixture lives on this platform.
applyResourceTable(
  RESOURCE_TABLE,
  'desktop',
  (dep, definition) => resources.define(dep, definition as never),
  (name) => (MODEL_CONSTANTS as Record<string, unknown>)[name],
  (kind, file) => resolveTableAsset(kind, file)
)

function readJsonConfig(configPath: string) {
  return JSON.parse(fs.readFileSync(configPath, 'utf8')) as Record<string, unknown>
}

// Exercises registryDownloadMaxRetries + registryStreamTimeoutMs end-to-end (see config-tests.ts).
function ensureDesktopE2EConfig() {
  const fixturePath = path.resolve(process.cwd(), 'fixtures/qvac.config.e2e.json')
  const existingPath = process.env['QVAC_CONFIG_PATH']
  const fixtureConfig = readJsonConfig(fixturePath)
  const existingConfig = existingPath ? readJsonConfig(existingPath) : {}
  const mergedConfig = {
    ...fixtureConfig,
    ...existingConfig
  }
  const configuredPlugins = Array.isArray(mergedConfig['plugins'])
    ? mergedConfig['plugins'].filter((plugin): plugin is string => typeof plugin === 'string')
    : []
  const desktopConfig = {
    ...mergedConfig,
    plugins: Array.from(new Set([...configuredPlugins, '@qvac/sdk/audiogen-ggml/plugin']))
  }
  const generatedPath = path.resolve(process.cwd(), 'qvac.config.e2e.generated.json')

  fs.writeFileSync(generatedPath, `${JSON.stringify(desktopConfig, null, 2)}\n`)
  process.env['QVAC_CONFIG_PATH'] = generatedPath

  if (existingPath) {
    console.log(
      `📦 Desktop e2e config merged ${fixturePath} with ${existingPath}; using ${generatedPath}`
    )
  } else {
    console.log(`📦 Desktop e2e config set to ${generatedPath}`)
  }
}

function resolveBatchAttachmentPath(inputPath: string) {
  const fileName = inputPath.split('/').pop()
  if (!fileName) return inputPath
  return path.resolve(process.cwd(), 'assets/images', fileName)
}

export async function bootstrap(filteredTests?: TestDefinition[]) {
  ensureDesktopE2EConfig()

  // `filteredTests` (when present) is the producer's post-filter test list
  // delivered via register-ack; absence keeps the legacy "warm everything" path.
  const allowedDeps = filteredTests ? collectTestDeps(filteredTests) : undefined
  await resources.downloadAllOnce(console.log, { allowedDeps })
}

const stepBindings = createStepBindings(resources)

export const executor = createExecutor({
  handlers: [
    new SkipExecutor(
      /^snap-storage-/,
      'Snap storage tests require the strict-confined Snap consumer'
    ),
    new ModelLoadingExecutor(resources),
    new BatchCompletionExecutor(resources, {
      resolveAttachmentPath: resolveBatchAttachmentPath
    }),
    new CompletionExecutor(resources),
    new TranscriptionExecutor(resources),
    new TranscribeStreamEventsExecutor(resources),
    new EmbeddingExecutor(resources),
    new RagExecutor(resources),
    new VectorIndexExecutor(resources),
    new ModelInfoExecutor(resources),
    new WrongModelExecutor(resources),
    new ErrorExecutor(resources),
    new ToolsExecutor(resources),

    // Must precede TranslationExecutor — patterns overlap, dispatch is first-match-wins.
    new TranslationBergamotCacheExecutor(),
    new TranslationExecutor(resources),
    new ShardedModelExecutor(resources),
    new OcrExecutor(resources),
    new VlaExecutor(resources),
    new ClassificationExecutor(resources),
    new TtsExecutor(resources),
    new ConfigReloadExecutor(resources),
    new NodeLoggingExecutor(resources),
    new RegistryExecutor(resources),
    new HttpEmbeddingExecutor(resources),
    new KvCacheExecutor(resources),
    new ParakeetStreamExecutor(resources),
    new ParakeetExecutor(resources),
    new BciExecutor(resources),
    new VisionExecutor(resources),
    // Must precede DownloadExecutor — its /^download-/ pattern also matches
    // download-resilience-*, and dispatch is first-match-wins.
    new DownloadResilienceExecutor(),
    new DownloadExecutor(),
    new NodeDiffusionExecutor(resources),
    new NodeWorldExecutor(resources),
    new AudioGenExecutor(resources, {
      resolveAudioAsset: (fileName) => path.resolve(process.cwd(), 'assets/audio', fileName)
    }),
    new FinetuneExecutor(resources),
    new LifecycleExecutor(resources),
    new SystemResourcesExecutor(),
    new ConfigExecutor(),
    new NoLingeringBareExecutor(),
    new KvCacheRestartExecutor(resources),
    new MultiGpuExecutor(resources),
    new NodeCancellationExecutor(resources),
    new PluginExecutor(resources)
  ],
  profiling: {
    init: () => profiler.enable({ mode: 'summary', includeServerBreakdown: true }),
    exportData: () => profiler.exportJSON()
  }
})

// A definition carrying `steps` is run by the shared interpreter instead of the
// executor above. That is what makes JS the reference implementation rather
// than merely the first one: the same interpreter, over the same catalog, as
// every other client. Definitions without `steps` are untouched.
executor.stepBindings = stepBindings
