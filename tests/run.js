#!/usr/bin/env node
/* Headless test runner: loads the engine files into a fake `window` and runs tests/suite.js. */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

globalThis.window = globalThis;
const root = path.join(__dirname, '..');
for (const f of ['nn', 'data', 'methods', 'merge', 'lab', 'advisor', 'share', 'mkimport']) {
  const file = path.join(root, 'js', `${f}.js`);
  vm.runInThisContext(fs.readFileSync(file, 'utf8'), { filename: file });
}
const suite = require('./suite.js');

const t0 = Date.now();
const results = suite(globalThis.MM);
let failed = 0;
for (const r of results) {
  if (!r.ok) failed++;
  console.log(`${r.ok ? '  ✓' : '  ✗'} ${r.name}${r.detail ? `  — ${r.detail}` : ''}`);
}
console.log(`\n${results.length - failed}/${results.length} passed in ${((Date.now() - t0) / 1000).toFixed(1)}s`);
process.exit(failed ? 1 : 0);
