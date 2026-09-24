/**
 * Checks a release train branch describes what it carries.
 *
 * The train is the six packages that depend on each other in a line, from
 * @qvac/inference through to the two plugins. That is not the SDK pod's
 * package roster — see .github/teams/sdk.json, which also owns
 * registry-server, rag, logging, error and test-suite. Those release on their
 * own.
 *
 * The per-package guard (.github/actions/release-merge-guard) compares one
 * branch version against one package.json. A train moves all six at once, on
 * their own numbers, so that comparison has nothing to anchor to. What anchors
 * here is the engine: @qvac/inference and @qvac/sdk share a major and minor,
 * packages/sdk lint enforces it, and the branch carries that number.
 */

export const TRAIN_PACKAGES = [
  { slug: 'inference', dir: 'packages/inference' },
  { slug: 'sdk', dir: 'packages/sdk' },
  { slug: 'cli', dir: 'packages/cli' },
  { slug: 'ai-sdk-provider', dir: 'packages/ai-sdk-provider' },
  { slug: 'opencode-plugin', dir: 'plugins/opencode' },
  { slug: 'openclaw-plugin', dir: 'plugins/openclaw' },
]

export const BRANCH_PATTERN = /^release-train-(\d+\.\d+\.\d+)$/

const ZERO_SHA = '0000000000000000000000000000000000000000'

export function parseBranch (ref) {
  const match = BRANCH_PATTERN.exec(ref)
  return match ? match[1] : null
}

function majorMinor (version) {
  const parts = version.split('.')
  return `${parts[0]}.${parts[1]}`
}

/**
 * @param {object} io
 * @param {(path: string) => string} io.readManifest   package.json at head
 * @param {() => string[]} io.changedFiles             paths changed in this push
 */
export function checkReleaseTrain (ref, baseSha, io) {
  const errors = []
  const branchVersion = parseBranch(ref)

  if (!branchVersion) {
    errors.push(
      `Invalid release train branch name — expected release-train-x.y.z, actual: ${ref}`
    )
    return errors
  }

  const versions = new Map()
  for (const pkg of TRAIN_PACKAGES) {
    const manifestPath = `${pkg.dir}/package.json`
    let manifest
    try {
      manifest = JSON.parse(io.readManifest(manifestPath))
    } catch (err) {
      errors.push(`Could not read ${manifestPath}: ${err.message}`)
      continue
    }
    versions.set(pkg.slug, manifest.version)
  }

  // The branch names the engine version, so both engine packages must be at it.
  for (const slug of ['inference', 'sdk']) {
    const version = versions.get(slug)
    if (version && version !== branchVersion) {
      errors.push(
        `Engine version mismatch — branch says ${branchVersion}, ${slug} package.json says ${version}`
      )
    }
  }

  // Independent of the branch: lint rejects an sdk whose engine range crosses
  // a minor, and a release that ships them apart only fails later and louder.
  const inference = versions.get('inference')
  const sdk = versions.get('sdk')
  if (inference && sdk && majorMinor(inference) !== majorMinor(sdk)) {
    errors.push(
      `@qvac/inference ${inference} and @qvac/sdk ${sdk} must share a major and minor`
    )
  }

  // Initial branch push has no diff to inspect.
  if (!baseSha || baseSha === ZERO_SHA) {
    return errors
  }

  // A package that moved must say why it moved. Ones the cascade did not reach
  // are not in this release and are not asked for a changelog.
  const changed = new Set(io.changedFiles())
  for (const pkg of TRAIN_PACKAGES) {
    const manifestPath = `${pkg.dir}/package.json`
    if (!changed.has(manifestPath)) continue
    const changelogPath = `${pkg.dir}/CHANGELOG.md`
    if (!changed.has(changelogPath)) {
      errors.push(
        `${pkg.slug} changed version but not its changelog — ${changelogPath} is untouched`
      )
    }
  }

  return errors
}
