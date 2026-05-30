const profileSelect = document.querySelector('#profile');
const formFields = document.querySelector('#formFields');
const output = document.querySelector('#output');
const statusEl = document.querySelector('#status');
const notesEl = document.querySelector('#notes');

let currentDefaults = {};
let lastCombinedShell = '';

function isLongField(key, value) {
  return ['extra_vllm_args', 'volumes', 'envs', 'ulimits', 'proxy_unsets'].includes(key) || String(value || '').includes('\n');
}

function renderForm(defaults) {
  currentDefaults = defaults;
  formFields.innerHTML = '';
  Object.entries(defaults).forEach(([key, value]) => {
    if (key === 'items') return;
    const wrapper = document.createElement('div');
    wrapper.className = 'field';
    const label = document.createElement('label');
    label.textContent = key;
    label.htmlFor = `field-${key}`;
    const input = isLongField(key, value) ? document.createElement('textarea') : document.createElement('input');
    input.id = `field-${key}`;
    input.name = key;
    input.value = Array.isArray(value) ? value.join('\n') : String(value ?? '');
    wrapper.append(label, input);
    formFields.appendChild(wrapper);
  });
}

function collectPayload() {
  const payload = { profile: profileSelect.value };
  formFields.querySelectorAll('input, textarea').forEach((input) => {
    payload[input.name] = input.value;
  });
  return payload;
}

async function loadDefaults() {
  const response = await fetch(`/api/defaults?profile=${encodeURIComponent(profileSelect.value)}`);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'failed to load defaults');
  renderForm(data.defaults);
  notesEl.innerHTML = '';
  (data.issue_notes || []).forEach((note) => {
    const li = document.createElement('li');
    li.textContent = note;
    notesEl.appendChild(li);
  });
  statusEl.textContent = `已加载 ${data.profile} 默认值`;
}

async function generateCommand() {
  const response = await fetch('/api/command', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(collectPayload()),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'failed to generate command');
  lastCombinedShell = data.combined_shell || data.shell_command || '';
  output.textContent = lastCombinedShell || JSON.stringify(data, null, 2);
  statusEl.textContent = `已生成 ${data.profile}，executed=${data.executed}`;
}

async function copyCommand() {
  await navigator.clipboard.writeText(lastCombinedShell || output.textContent);
  statusEl.textContent = '已复制';
}

document.querySelector('#loadDefaults').addEventListener('click', () => loadDefaults().catch((err) => { statusEl.textContent = err.message; }));
document.querySelector('#generate').addEventListener('click', () => generateCommand().catch((err) => { statusEl.textContent = err.message; }));
document.querySelector('#copy').addEventListener('click', () => copyCommand().catch((err) => { statusEl.textContent = err.message; }));
profileSelect.addEventListener('change', () => loadDefaults().catch((err) => { statusEl.textContent = err.message; }));

loadDefaults().catch((err) => { statusEl.textContent = err.message; });
