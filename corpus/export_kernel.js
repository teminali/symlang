'use strict';
/* Loads the Deterministic Coder Kernel browser scripts in a Node vm context (the same way
 * test/kernel_presets.test.js does: classic scripts that attach to a global `DC`) and prints
 * one JSON object on stdout: { presets, primer, schema }.
 * Usage: node export_kernel.js <path to deterministic-coder/ui/public/backend> */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const backend = process.argv[2];
if (!backend) { console.error('usage: node export_kernel.js <backend dir>'); process.exit(2); }

const sandbox = { console, performance };
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
for (const f of ['kernel_expr.js', 'kernel_core.js', 'kernel_presets.js', 'kernel_model.js']) {
  vm.runInContext(fs.readFileSync(path.join(backend, f), 'utf8'), sandbox, { filename: f });
}
const DC = sandbox.DC;
const plain = (v) => JSON.parse(JSON.stringify(v));
const out = {
  presets: plain(DC.kernel.presets),
  primer: DC.kernelModel.primer('local'),
  schema: plain(DC.kernelModel.schema),
};
process.stdout.write(JSON.stringify(out));
