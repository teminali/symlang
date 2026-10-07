// Dumps the six Deterministic Coder kernel preset specs to JSON on stdout.
// kernel_presets.js is a classic browser script that attaches to a global (DC.kernel.presets).
// Same loading idea as deterministic-coder/test/kernel_presets.test.js: run it with vm in a fake global.
// DC.kernel only needs to exist (presets are plain data), so we stub it instead of loading kernel_core.js.
'use strict';
const fs = require('fs');
const vm = require('vm');
const src = process.argv[2];
if (!src) { console.error('usage: node dump_presets.js <path to kernel_presets.js>'); process.exit(2); }
const sandbox = { DC: { kernel: {} } };
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(src, 'utf8'), sandbox, { filename: 'kernel_presets.js' });
const presets = JSON.parse(JSON.stringify(sandbox.DC.kernel.presets));
const out = {};
for (const k of Object.keys(presets).sort()) out[k] = presets[k];
process.stdout.write(JSON.stringify(out, null, 1) + '\n');
