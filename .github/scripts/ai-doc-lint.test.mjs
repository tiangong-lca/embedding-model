import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';

import {
  collectExpectedDocs,
  detectImpactLayout,
  globToRegExp,
  isKeyMarkdownDoc,
  loadImpactFiles,
  matchRules,
  missingMarkdownMetadata,
  missingYamlMetadata,
  normalizePath,
} from './ai-doc-lint.mjs';

test('normalizePath and globToRegExp support repo-relative matching', () => {
  assert.equal(normalizePath('.\\tiangong-lca-next\\config\\routes.ts'), 'tiangong-lca-next/config/routes.ts');
  assert.equal(globToRegExp('tiangong-lca-next/**').test('tiangong-lca-next/config/routes.ts'), true);
  assert.equal(globToRegExp('ai/*.md').test('ai/quality-rubric.md'), true);
  assert.equal(globToRegExp('ai/*.md').test('ai/nested/file.md'), false);
});

test('detectImpactLayout distinguishes workspace and repo roots', () => {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), 'ai-doc-lint-layout-'));
  mkdirSync(path.join(tempDir, 'ai'), { recursive: true });

  assert.equal(detectImpactLayout(tempDir), 'none');

  writeFileSync(path.join(tempDir, 'ai', 'doc-impact.yaml'), JSON.stringify({ version: 1 }));
  assert.equal(detectImpactLayout(tempDir), 'repo');

  writeFileSync(path.join(tempDir, 'ai', 'doc-impact-map.yaml'), JSON.stringify({ version: 1 }));
  assert.equal(detectImpactLayout(tempDir), 'workspace');
});

test('isKeyMarkdownDoc excludes YAML contract files', () => {
  assert.equal(isKeyMarkdownDoc('ai/quality-rubric.md'), true);
  assert.equal(isKeyMarkdownDoc('AGENTS.md'), true);
  assert.equal(isKeyMarkdownDoc('ai/workspace.yaml'), false);
});

test('missingMarkdownMetadata detects absent frontmatter keys', () => {
  const text = `---
title: Example
docType: contract
scope: workspace
status: draft
authoritative: false
owner: lca-workspace
language: en
whenToUse:
  - x
whenToUpdate:
  - y
checkPaths:
  - ai/**
lastReviewedAt: 2026-04-18
---

# Example
`;

  assert.deepEqual(missingMarkdownMetadata(text), ['lastReviewedCommit']);
});

test('missingYamlMetadata detects absent top-level review fields', () => {
  const text = JSON.stringify({ version: 1, lastReviewedAt: '2026-04-18' });
  assert.deepEqual(missingYamlMetadata(text, 'ai/example.yaml'), ['lastReviewedCommit']);
});

test('loadImpactFiles, matchRules, and collectExpectedDocs resolve repo-local paths', () => {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), 'ai-doc-lint-'));
  mkdirSync(path.join(tempDir, 'ai'), { recursive: true });
  mkdirSync(path.join(tempDir, 'subrepo', 'ai'), { recursive: true });

  writeFileSync(
    path.join(tempDir, 'ai', 'doc-impact-map.yaml'),
    JSON.stringify(
      {
        version: 1,
        lastReviewedAt: '2026-04-18',
        lastReviewedCommit: 'abc',
        rules: [
          {
            id: 'root-rule',
            scope: 'workspace',
            repo: 'lca-workspace',
            triggers: [{ path: 'AGENTS.md', kind: 'doc-contract' }],
            requiredDocs: [{ path: 'ai/workspace.yaml', mode: 'review_or_update' }],
            reason: 'root',
          },
        ],
      },
      null,
      2,
    ),
  );

  writeFileSync(
    path.join(tempDir, 'subrepo', 'ai', 'doc-impact.yaml'),
    JSON.stringify(
      {
        version: 1,
        lastReviewedAt: '2026-04-18',
        lastReviewedCommit: 'abc',
        rules: [
          {
            id: 'repo-rule',
            scope: 'repo',
            repo: 'subrepo',
            triggers: [{ path: 'src/**', kind: 'code' }],
            requiredDocs: [{ path: 'ai/task-router.md', mode: 'review_or_update' }],
            reason: 'repo',
          },
        ],
      },
      null,
      2,
    ),
  );

  const loadedRules = loadImpactFiles(tempDir);
  const matches = matchRules(['AGENTS.md', 'subrepo/src/index.ts'], loadedRules);
  const expectedDocs = collectExpectedDocs(matches);

  assert.equal(loadedRules.length, 2);
  assert.equal(matches.length, 2);
  assert.deepEqual([...expectedDocs.keys()].sort(), ['ai/workspace.yaml', 'subrepo/ai/task-router.md']);
});

test('loadImpactFiles supports repo-root mode', () => {
  const tempDir = mkdtempSync(path.join(os.tmpdir(), 'ai-doc-lint-repo-'));
  mkdirSync(path.join(tempDir, 'ai'), { recursive: true });

  writeFileSync(
    path.join(tempDir, 'ai', 'doc-impact.yaml'),
    JSON.stringify(
      {
        version: 1,
        lastReviewedAt: '2026-04-18',
        lastReviewedCommit: 'abc',
        rules: [
          {
            id: 'repo-rule',
            scope: 'repo',
            repo: 'example',
            triggers: [{ path: 'src/**', kind: 'code' }],
            requiredDocs: [{ path: 'ai/validation.md', mode: 'review_or_update' }],
            reason: 'repo-root',
          },
        ],
      },
      null,
      2,
    ),
  );

  const loadedRules = loadImpactFiles(tempDir);
  const matches = matchRules(['src/index.ts'], loadedRules);
  const expectedDocs = collectExpectedDocs(matches);

  assert.equal(loadedRules.length, 1);
  assert.equal(matches.length, 1);
  assert.deepEqual([...expectedDocs.keys()], ['ai/validation.md']);
});

// Trace the real workflow commands and shell gate. The fake node terminates at
// the process boundary rather than recursively executing this same test suite.
function traceDocGate(t, { unitStatus = 0, lintStatus = 0, base = 'HEAD', workflow = false } = {}) {
  const repo = fileURLToPath(new URL('../..', import.meta.url));
  const temp = mkdtempSync(path.join(os.tmpdir(), 'ai-doc-gate-trace-'));
  t.after(() => rmSync(temp, { recursive: true, force: true }));
  const trace = path.join(temp, 'trace');
  const transport = `
node() {
  printf '%s\n' "$*" >> "$GATE_TRACE"
  if [ "$1" = "--test" ]; then return "$UNIT_STATUS"; fi
  return "$LINT_STATUS"
}
git() {
  case "$*" in
    'rev-parse --show-toplevel') printf '%s\n' "$GATE_REPO" ;;
    'merge-base HEAD HEAD'|'rev-parse HEAD') printf '%040d\n' 1 ;;
    'merge-base HEAD refs/heads/ai-doc-test-missing-base') return 1 ;;
    *) return 90 ;;
  esac
}
`;
  const env = { ...process.env, GATE_REPO: repo, GATE_TRACE: trace,
    UNIT_STATUS: String(unitStatus), LINT_STATUS: String(lintStatus), BASE_REF: base,
    DOCPACT_HEAD_REF: 'HEAD' };
  const commands = workflow
    ? [...readFileSync(path.join(repo, '.github/workflows/ai-doc-lint.yml'), 'utf8')
      .matchAll(/^\s+run: (.+)$/gm)].map((match) => match[1])
      .filter((command) => command.includes('ai-doc-lint'))
    : ['scripts/ai-doc-lint-gate.sh --base "$BASE_REF"'];
  assert.ok(commands.length > 0, 'workflow must invoke doc validation');
  let status;
  for (const command of commands) {
    // Source the unchanged gate in this shell so transport functions apply to
    // its actual argv/error paths without fake executables or recursive tests.
    let executable = command;
    if (command.startsWith('scripts/ai-doc-lint-gate.sh')) {
      assert.equal(command, 'scripts/ai-doc-lint-gate.sh --base "$BASE_REF"');
      executable = 'set -- --base "$BASE_REF"; . scripts/ai-doc-lint-gate.sh';
    }
    const shellArgs = workflow ? ['-e', '-c'] : ['-c'];
    const result = spawnSync('sh', [...shellArgs, transport + executable], { cwd: repo, env, encoding: 'utf8' });
    assert.ifError(result.error);
    status = result.status;
    if (status !== 0) break;
  }
  let calls = [];
  try { calls = readFileSync(trace, 'utf8').trim().split('\n'); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
  return { status, calls };
}

test('manual workflow invokes unit then enforced lint exactly once through its shell gate', (t) => {
  const { status, calls } = traceDocGate(t, { workflow: true });
  assert.equal(status, 0);
  assert.equal(calls.length, 2, `unexpected validation calls: ${calls.join(' -> ')}`);
  assert.equal(calls[0], '--test .github/scripts/ai-doc-lint.test.mjs');
  assert.match(calls[1], /^\.github\/scripts\/ai-doc-lint\.mjs --mode enforce --base [a-f0-9]{40} --head [a-f0-9]{40}$/);
});

test('shell gate stops after failed unit tests, propagates lint failures and rejects missing base', (t) => {
  const unitFailure = traceDocGate(t, { unitStatus: 7 });
  assert.equal(unitFailure.status, 7);
  assert.deepEqual(unitFailure.calls, ['--test .github/scripts/ai-doc-lint.test.mjs']);
  const lintFailure = traceDocGate(t, { lintStatus: 9 });
  assert.equal(lintFailure.status, 9);
  assert.equal(lintFailure.calls.length, 2);
  const missingBase = traceDocGate(t, { base: 'refs/heads/ai-doc-test-missing-base' });
  assert.equal(missingBase.status, 2);
  assert.deepEqual(missingBase.calls, []);
});
