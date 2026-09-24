// The release train branch carries one version for six packages, so the guard
// has to decide which mismatches are releases in progress and which are
// mistakes. These lock that line.
import test from 'node:test'
import assert from 'node:assert/strict'
import { checkReleaseTrain, parseBranch } from '../lib/release-train-guard.mjs'

const AT_0_21_0 = {
  'packages/inference/package.json': '0.21.0',
  'packages/sdk/package.json': '0.21.0',
  'packages/cli/package.json': '0.15.0',
  'packages/ai-sdk-provider/package.json': '0.8.1',
  'plugins/opencode/package.json': '0.3.3',
  'plugins/openclaw/package.json': '0.3.3',
}

function io(versions, changed = []) {
  return {
    readManifest: (path) => {
      const version = versions[path]
      if (!version) throw new Error(`no fixture for ${path}`)
      return JSON.stringify({ version })
    },
    changedFiles: () => changed,
  }
}

const INITIAL_PUSH = ''

test('parses the engine version out of the branch name', () => {
  assert.equal(parseBranch('release-train-0.21.0'), '0.21.0')
  assert.equal(parseBranch('release-train-0.21.0-rc1'), null)
  assert.equal(parseBranch('release-sdk-0.21.0'), null)
  assert.equal(parseBranch('release-train'), null)
})

test('accepts a train at the branch version', () => {
  const errors = checkReleaseTrain('release-train-0.21.0', INITIAL_PUSH, io(AT_0_21_0))
  assert.deepEqual(errors, [])
})

test('rejects a branch name that is not release-train-x.y.z', () => {
  const errors = checkReleaseTrain('release-train', INITIAL_PUSH, io(AT_0_21_0))
  assert.equal(errors.length, 1)
  assert.match(errors[0], /Invalid release train branch name/)
})

test('rejects an engine package that is not at the branch version', () => {
  const versions = { ...AT_0_21_0, 'packages/sdk/package.json': '0.21.1' }
  const errors = checkReleaseTrain('release-train-0.21.0', INITIAL_PUSH, io(versions))
  assert.equal(errors.length, 1)
  assert.match(errors[0], /branch says 0\.21\.0, sdk package\.json says 0\.21\.1/)
})

test('rejects an engine split across a minor', () => {
  const versions = {
    ...AT_0_21_0,
    'packages/inference/package.json': '0.22.0',
    'packages/sdk/package.json': '0.22.0',
  }
  const errors = checkReleaseTrain('release-train-0.22.0', INITIAL_PUSH, io(versions))
  assert.deepEqual(errors, [])

  const split = { ...versions, 'packages/sdk/package.json': '0.21.0' }
  const errorsSplit = checkReleaseTrain('release-train-0.22.0', INITIAL_PUSH, io(split))
  assert.equal(errorsSplit.length, 2)
  assert.match(errorsSplit[1], /must share a major and minor/)
})

test('does not ask the agent stack to match the branch version', () => {
  // cli 0.15.0 on a 0.21.0 branch is the normal case, not a mismatch.
  const errors = checkReleaseTrain('release-train-0.21.0', INITIAL_PUSH, io(AT_0_21_0))
  assert.deepEqual(errors, [])
})

test('requires a changelog from every package whose manifest moved', () => {
  const changed = [
    'packages/inference/package.json',
    'packages/inference/CHANGELOG.md',
    'packages/cli/package.json',
  ]
  const errors = checkReleaseTrain('release-train-0.21.0', 'abc123', io(AT_0_21_0, changed))
  assert.equal(errors.length, 1)
  assert.match(errors[0], /cli changed version but not its changelog/)
})

test('asks no changelog of a package the release did not touch', () => {
  const changed = ['packages/inference/package.json', 'packages/inference/CHANGELOG.md']
  const errors = checkReleaseTrain('release-train-0.21.0', 'abc123', io(AT_0_21_0, changed))
  assert.deepEqual(errors, [])
})

test('skips the changelog check on the initial branch push', () => {
  const errors = checkReleaseTrain(
    'release-train-0.21.0',
    '0000000000000000000000000000000000000000',
    io(AT_0_21_0, [])
  )
  assert.deepEqual(errors, [])
})
