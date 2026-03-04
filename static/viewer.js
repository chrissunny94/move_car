async function fetchTokens() {
  const status = document.getElementById('status');
  try {
    const resp = await fetch('/npz/list');
    if (!resp.ok) {
      status.textContent = 'Failed to load token list: ' + resp.status;
      return;
    }
    const data = await resp.json();
    const select = document.getElementById('token-select');

    // clear existing options except the first placeholder
    while (select.options.length > 1) {
      select.remove(1);
    }

    for (const t of data.tokens || []) {
      const opt = document.createElement('option');
      opt.value = t;
      opt.textContent = t;
      select.appendChild(opt);
    }

    if (!data.tokens || data.tokens.length === 0) {
      status.textContent = 'No NPZ files found.';
    } else {
      status.textContent = 'Loaded ' + data.tokens.length + ' samples.';
    }
  } catch (e) {
    status.textContent = 'Error loading token list: ' + e;
  }
}

async function loadOcc() {
  const select = document.getElementById('token-select');
  const token = select.value;
  if (!token) return;
  const status = document.getElementById('status');
  status.textContent = 'Loading ' + token + '...';
  try {
    const resp = await fetch('/npz/' + token);
    if (!resp.ok) {
      const txt = await resp.text();
      status.textContent = 'Error ' + resp.status + ': ' + txt;
      return;
    }
    const data = await resp.json();
    status.textContent = 'Loaded ' + data.num_occupied + ' occupied cells';
    drawOcc(data);
  } catch (e) {
    status.textContent = 'Error: ' + e;
  }
}

function drawOcc(data) {
  const canvas = document.getElementById('canvas');
  const ctx = canvas.getContext('2d');
  const shape = data.bev_shape;
  const H = shape[0];
  const W = shape[1];
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const scaleX = canvas.width / W;
  const scaleY = canvas.height / H;

  // color for occupied cells
  ctx.fillStyle = 'red';

  // support both full and sampled responses
  const indices = data.occupied_indices || data.occupied_indices_sample || [];
  for (const ij of indices) {
    const i = ij[0];
    const j = ij[1];
    const x = j * scaleX;
    const y = i * scaleY;
    ctx.fillRect(x, y, scaleX, scaleY);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('load-btn').addEventListener('click', loadOcc);
  fetchTokens();
});