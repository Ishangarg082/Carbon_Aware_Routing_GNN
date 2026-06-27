"""
HTML animation generator for Carbon-Aware Routing visualization.

Generates a self-contained interactive dashboard with:
- Network graph visualization
- Real-time carbon emissions chart
- Simulation playback controls
- Traffic load display on nodes
"""

import json
import os
import numpy as np


def generate_animation(topology, carbon_manager, ca_results, bl_results,
                       output_path="results/simulation_animation.html"):
    graph = topology['graph']
    positions = topology['positions']
    num_nodes = graph.number_of_nodes()

    nodes = []
    for i in range(num_nodes):
        x, y = positions.get(i, (0.5, 0.5))
        profile = carbon_manager.node_assignments.get(i, 'mixed_grid')
        nodes.append({
            'id': i,
            'x': round(x * 800 + 80, 1),
            'y': round((1 - y) * 500 + 80, 1),
            'profile': profile
        })

    edges = []
    for u, v in graph.edges():
        edges.append([int(u), int(v)])

    ca_carbon = ca_results.get('carbon_history', [])
    bl_carbon = bl_results.get('carbon_history', [])
    ts = ca_results.get('timestamps', [i * 3600 for i in range(len(ca_carbon))])

    traffic = []
    for entry in ca_results.get('routing_history', []):
        loads = entry.get('traffic_loads', {})
        traffic.append({str(k): round(v, 2) for k, v in loads.items()})

    data = {
        'nodes': nodes, 'edges': edges,
        'hours': [round(t / 3600, 1) for t in ts[:len(ca_carbon)]],
        'ca': [round(c, 1) for c in ca_carbon],
        'bl': [round(c, 1) for c in bl_carbon[:len(ca_carbon)]],
        'traffic': traffic
    }

    total_ca = sum(ca_carbon)
    total_bl = sum(bl_carbon[:len(ca_carbon)])
    pct = ((total_bl - total_ca) / total_bl * 100) if total_bl > 0 else 0
    saved = total_bl - total_ca

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Carbon-Aware Routing - Simulation Results</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root {{
  --bg: #ffffff;
  --bg-secondary: #f9fafb;
  --bg-hover: #f3f4f6;
  --border: #e5e7eb;
  --border-dark: #d1d5db;
  --text: #111827;
  --text-secondary: #6b7280;
  --text-muted: #9ca3af;
  --blue: #2563eb;
  --blue-light: #dbeafe;
  --green: #16a34a;
  --green-light: #dcfce7;
  --red: #dc2626;
  --red-light: #fee2e2;
  --amber: #d97706;
  --node-solar: #16a34a;
  --node-wind: #2563eb;
  --node-hydro: #0891b2;
  --node-coal: #dc2626;
  --node-nuclear: #7c3aed;
  --node-mixed: #6b7280;
}}

body {{
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  background: var(--bg);
  color: var(--text);
  overflow: hidden;
  height: 100vh;
}}

/* ---- Top bar ---- */
.topbar {{
  height: 52px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  border-bottom: 1px solid var(--border);
  background: var(--bg);
}}
.topbar .brand {{
  font-weight: 700;
  font-size: 15px;
  color: var(--text);
}}
.topbar .meta {{
  font-size: 12px;
  color: var(--text-secondary);
  display: flex;
  gap: 24px;
}}
.topbar .meta .val {{
  color: var(--text);
  font-weight: 600;
}}

/* ---- Main layout ---- */
.main {{
  display: flex;
  height: calc(100vh - 52px);
}}

/* ---- Canvas area ---- */
.canvas-wrap {{
  flex: 1;
  position: relative;
  background: var(--bg-secondary);
  overflow: hidden;
  border-right: 1px solid var(--border);
}}
.canvas-wrap canvas {{
  display: block;
  width: 100%;
  height: 100%;
}}
.canvas-label {{
  position: absolute;
  top: 14px;
  left: 18px;
  font-size: 11px;
  font-weight: 600;
  color: var(--text-secondary);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}}

/* ---- Inline legend on canvas ---- */
.canvas-legend {{
  position: absolute;
  bottom: 16px;
  left: 18px;
  display: flex;
  gap: 14px;
  font-size: 11px;
  color: var(--text-secondary);
}}
.canvas-legend .item {{
  display: flex;
  align-items: center;
  gap: 5px;
}}
.canvas-legend .dot {{
  width: 8px;
  height: 8px;
  border-radius: 50%;
}}

/* ---- Sidebar ---- */
.sidebar {{
  width: 320px;
  background: var(--bg);
  overflow-y: auto;
  display: flex;
  flex-direction: column;
}}
.sidebar::-webkit-scrollbar {{ width: 4px; }}
.sidebar::-webkit-scrollbar-thumb {{ background: var(--border-dark); border-radius: 2px; }}

.panel {{
  padding: 16px 20px;
  border-bottom: 1px solid var(--border);
}}
.panel-title {{
  font-size: 11px;
  font-weight: 600;
  color: var(--text-muted);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-bottom: 12px;
}}

/* ---- Result block ---- */
.result-number {{
  font-size: 36px;
  font-weight: 700;
  letter-spacing: -0.02em;
  line-height: 1;
}}
.result-number.positive {{ color: var(--green); }}
.result-number.negative {{ color: var(--red); }}
.result-sub {{
  font-size: 12px;
  color: var(--text-secondary);
  margin-top: 4px;
}}
.result-row {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 6px 0;
  font-size: 13px;
}}
.result-row .label {{ color: var(--text-secondary); }}
.result-row .value {{ font-weight: 600; font-variant-numeric: tabular-nums; }}
.result-row .value.green {{ color: var(--green); }}
.result-row .value.red {{ color: var(--red); }}
.result-divider {{
  height: 1px;
  background: var(--border);
  margin: 10px 0;
}}

/* ---- Playback ---- */
.playback-controls {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 10px;
}}
.play-btn {{
  width: 32px;
  height: 32px;
  border-radius: 6px;
  border: 1px solid var(--border-dark);
  background: var(--bg);
  color: var(--text);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  transition: background 0.15s, border-color 0.15s;
}}
.play-btn:hover {{ background: var(--bg-hover); border-color: var(--blue); }}
.play-btn.active {{ background: var(--blue-light); border-color: var(--blue); color: var(--blue); }}
.reset-btn {{
  height: 28px;
  padding: 0 10px;
  border-radius: 4px;
  border: 1px solid var(--border-dark);
  background: var(--bg);
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 11px;
  font-family: inherit;
  transition: background 0.15s;
}}
.reset-btn:hover {{ background: var(--bg-hover); color: var(--text); }}
.speed-select {{
  height: 28px;
  padding: 0 8px;
  border-radius: 4px;
  border: 1px solid var(--border-dark);
  background: var(--bg);
  color: var(--text-secondary);
  font-size: 11px;
  font-family: inherit;
  cursor: pointer;
  margin-left: auto;
}}

/* Custom range slider */
input[type=range] {{
  -webkit-appearance: none;
  width: 100%;
  height: 4px;
  border-radius: 2px;
  background: var(--border);
  outline: none;
  margin: 8px 0;
}}
input[type=range]::-webkit-slider-thumb {{
  -webkit-appearance: none;
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--blue);
  cursor: pointer;
  border: 2px solid white;
  box-shadow: 0 1px 3px rgba(0,0,0,0.15);
}}
input[type=range]::-moz-range-thumb {{
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--blue);
  cursor: pointer;
  border: 2px solid white;
}}
.time-label {{
  text-align: center;
  font-size: 11px;
  color: var(--text-muted);
  font-variant-numeric: tabular-nums;
}}

/* ---- Current hour stats ---- */
.hour-stat {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 5px 0;
  font-size: 12px;
}}
.hour-stat .dot {{
  width: 6px;
  height: 6px;
  border-radius: 50%;
  display: inline-block;
  margin-right: 6px;
}}
.hour-stat .lbl {{ color: var(--text-secondary); }}
.hour-stat .val {{
  font-weight: 600;
  font-variant-numeric: tabular-nums;
}}

/* ---- Chart panel ---- */
#chart {{ width: 100%; height: 140px; }}
.chart-legend {{
  display: flex;
  gap: 16px;
  justify-content: center;
  margin-top: 6px;
  font-size: 10px;
  color: var(--text-muted);
}}
.chart-legend .ln {{
  display: flex;
  align-items: center;
  gap: 4px;
}}
.chart-legend .swatch {{
  width: 16px;
  height: 2px;
  border-radius: 1px;
}}
.chart-legend .swatch.dashed {{
  background: repeating-linear-gradient(
    to right,
    var(--red) 0,
    var(--red) 4px,
    transparent 4px,
    transparent 7px
  );
}}

</style></head><body>

<div class="topbar">
  <div class="brand">Carbon-Aware Routing</div>
  <div class="meta">
    <div>{num_nodes} nodes &middot; {len(edges)} links</div>
    <div>Reduction <span class="val" style="color:var(--green)">{pct:.1f}%</span></div>
    <div>Saved <span class="val">{saved:,.0f} gCO&#8322;</span></div>
  </div>
</div>

<div class="main">
  <div class="canvas-wrap">
    <canvas id="net"></canvas>
    <div class="canvas-label">Network Topology</div>
    <div class="canvas-legend">
      <div class="item"><div class="dot" style="background:var(--node-solar)"></div>Solar</div>
      <div class="item"><div class="dot" style="background:var(--node-wind)"></div>Wind</div>
      <div class="item"><div class="dot" style="background:var(--node-hydro)"></div>Hydro</div>
      <div class="item"><div class="dot" style="background:var(--node-coal)"></div>Coal</div>
      <div class="item"><div class="dot" style="background:var(--node-nuclear)"></div>Nuclear</div>
      <div class="item"><div class="dot" style="background:var(--node-mixed)"></div>Mixed</div>
    </div>
  </div>

  <div class="sidebar">
    <!-- Result -->
    <div class="panel">
      <div class="panel-title">Carbon Reduction</div>
      <div class="result-number {'positive' if pct > 0 else 'negative'}">{pct:.1f}%</div>
      <div class="result-sub">{saved:,.0f} gCO&#8322; avoided over {len(data['hours'])} intervals</div>
      <div class="result-divider"></div>
      <div class="result-row">
        <span class="label">Baseline (shortest path)</span>
        <span class="value red">{total_bl:,.0f}</span>
      </div>
      <div class="result-row">
        <span class="label">GNN-optimized routing</span>
        <span class="value green">{total_ca:,.0f}</span>
      </div>
    </div>

    <!-- Playback -->
    <div class="panel">
      <div class="panel-title">Playback</div>
      <div class="playback-controls">
        <button class="play-btn" id="playBtn" onclick="toggle()" title="Play/Pause">&#9654;</button>
        <button class="reset-btn" onclick="step=0;up()">Reset</button>
        <select class="speed-select" id="spd" onchange="setSpd(this.value)">
          <option value="1200">0.5x</option>
          <option value="600" selected>1x</option>
          <option value="200">3x</option>
        </select>
      </div>
      <input type="range" id="sl" min="0" max="0" value="0" oninput="step=+this.value;up()">
      <div class="time-label" id="tm">0:00</div>
    </div>

    <!-- Current hour -->
    <div class="panel">
      <div class="panel-title">Current Interval</div>
      <div class="hour-stat">
        <span class="lbl"><span class="dot" style="background:var(--blue)"></span>GNN routing</span>
        <span class="val" id="cCA">&mdash;</span>
      </div>
      <div class="hour-stat">
        <span class="lbl"><span class="dot" style="background:var(--red)"></span>Baseline</span>
        <span class="val" id="cBL">&mdash;</span>
      </div>
      <div class="hour-stat">
        <span class="lbl">Savings</span>
        <span class="val" id="cSV" style="color:var(--green)">&mdash;</span>
      </div>
    </div>

    <!-- Chart -->
    <div class="panel" style="flex:1 0 auto;">
      <div class="panel-title">Emissions Over Time</div>
      <canvas id="chart"></canvas>
      <div class="chart-legend">
        <div class="ln"><div class="swatch" style="background:var(--blue)"></div>GNN</div>
        <div class="ln"><div class="swatch dashed"></div>Baseline</div>
      </div>
    </div>
  </div>
</div>

<script>
const D={json.dumps(data, separators=(',',':'))};

const COL={{
  solar_heavy:'#16a34a', wind_heavy:'#2563eb', coal_heavy:'#dc2626',
  nuclear:'#7c3aed', hydro:'#0891b2', mixed_grid:'#6b7280'
}};

let step=0, playing=false, timer=null, pkts=[];
const nc=document.getElementById('net'), ctx=nc.getContext('2d');
const cc=document.getElementById('chart'), cctx=cc.getContext('2d');

function resize() {{
  const dpr = devicePixelRatio || 1;
  nc.width = nc.clientWidth * dpr;
  nc.height = nc.clientHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  cc.width = cc.clientWidth * dpr;
  cc.height = 140 * dpr;
  cc.style.height = '140px';
  cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}}

function spawn() {{
  pkts = [];
  const sn = D.traffic[step] || {{}};
  D.edges.forEach(e => {{
    const ld = (sn[e[0]] || 0) + (sn[e[1]] || 0);
    const n = Math.min(Math.ceil(ld / 10), 2);
    for (let i = 0; i < n; i++)
      pkts.push({{ s: e[0], d: e[1], t: Math.random(), sp: 0.002 + Math.random() * 0.004 }});
  }});
}}

function drawNet() {{
  const W = nc.clientWidth, H = nc.clientHeight;
  const sx = W / 960, sy = H / 660;
  ctx.clearRect(0, 0, W, H);

  // Background
  ctx.fillStyle = '#f9fafb';
  ctx.fillRect(0, 0, W, H);

  // Subtle grid lines
  ctx.strokeStyle = '#e5e7eb';
  ctx.lineWidth = 0.5;
  for (let gx = 0; gx < W; gx += 40) {{
    ctx.beginPath();
    ctx.moveTo(gx, 0);
    ctx.lineTo(gx, H);
    ctx.stroke();
  }}
  for (let gy = 0; gy < H; gy += 40) {{
    ctx.beginPath();
    ctx.moveTo(0, gy);
    ctx.lineTo(W, gy);
    ctx.stroke();
  }}

  // Links
  const sn = D.traffic[step] || {{}};
  D.edges.forEach(e => {{
    const a = D.nodes[e[0]], b = D.nodes[e[1]];
    const ld = ((sn[e[0]] || 0) + (sn[e[1]] || 0));
    const intensity = Math.min(ld / 200, 1);
    ctx.strokeStyle = `rgba(107,114,128,${{0.15 + intensity * 0.25}})`;
    ctx.lineWidth = 1 + intensity * 1.5;
    ctx.beginPath();
    ctx.moveTo(a.x * sx, a.y * sy);
    ctx.lineTo(b.x * sx, b.y * sy);
    ctx.stroke();
  }});

  // Packets (small moving dots along edges)
  pkts.forEach(p => {{
    p.t += p.sp;
    if (p.t > 1) p.t -= 1;
    const a = D.nodes[p.s], b = D.nodes[p.d];
    const px = (a.x + (b.x - a.x) * p.t) * sx;
    const py = (a.y + (b.y - a.y) * p.t) * sy;

    ctx.beginPath();
    ctx.arc(px, py, 2, 0, Math.PI * 2);
    ctx.fillStyle = '#9ca3af';
    ctx.fill();
  }});

  // Nodes
  D.nodes.forEach(n => {{
    const x = n.x * sx, y = n.y * sy;
    const col = COL[n.profile] || '#6b7280';
    const load = sn[n.id] || 50;
    const r = 5 + Math.min(load / 100, 1) * 6;

    // Node circle with border
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = col;
    ctx.fill();
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Node label
    ctx.fillStyle = '#111827';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(n.id, x, y + r + 14);
  }});
}}

function drawChart() {{
  const W = cc.clientWidth, H = 140;
  cctx.clearRect(0, 0, W, H);
  if (!D.ca.length) return;

  const all = [...D.ca, ...D.bl];
  const mn = Math.min(...all) * 0.85, mx = Math.max(...all) * 1.1;
  const L = D.ca.length;
  const pad = {{ l: 44, r: 8, t: 8, b: 20 }};
  const cw = W - pad.l - pad.r, ch = H - pad.t - pad.b;
  const tx = i => pad.l + (i / (L - 1 || 1)) * cw;
  const ty = v => pad.t + (1 - (v - mn) / (mx - mn)) * ch;

  // Horizontal grid lines
  cctx.strokeStyle = '#e5e7eb';
  cctx.lineWidth = 1;
  for (let i = 0; i < 4; i++) {{
    const y = pad.t + ch / 3 * i;
    cctx.beginPath();
    cctx.moveTo(pad.l, y);
    cctx.lineTo(W - pad.r, y);
    cctx.stroke();
    const v = mx - (mx - mn) * i / 3;
    cctx.fillStyle = '#9ca3af';
    cctx.font = '9px Inter, sans-serif';
    cctx.textAlign = 'right';
    cctx.fillText(v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(0) + 'K' : v.toFixed(0), pad.l - 4, y + 3);
  }}

  // Baseline (dashed red line)
  cctx.setLineDash([4, 3]);
  cctx.strokeStyle = 'rgba(220,38,38,0.5)';
  cctx.lineWidth = 1.5;
  cctx.beginPath();
  for (let i = 0; i < L; i++) cctx[i ? 'lineTo' : 'moveTo'](tx(i), ty(D.bl[i]));
  cctx.stroke();

  // GNN line (solid blue)
  const up = Math.min(step, L - 1);
  cctx.setLineDash([]);

  // Area fill under GNN line
  if (up > 0) {{
    const areaGrad = cctx.createLinearGradient(0, pad.t, 0, H - pad.b);
    areaGrad.addColorStop(0, 'rgba(37,99,235,0.1)');
    areaGrad.addColorStop(1, 'rgba(37,99,235,0)');
    cctx.fillStyle = areaGrad;
    cctx.beginPath();
    cctx.moveTo(tx(0), H - pad.b);
    for (let i = 0; i <= up; i++) cctx.lineTo(tx(i), ty(D.ca[i]));
    cctx.lineTo(tx(up), H - pad.b);
    cctx.closePath();
    cctx.fill();
  }}

  // GNN line
  cctx.strokeStyle = '#2563eb';
  cctx.lineWidth = 2;
  cctx.beginPath();
  for (let i = 0; i <= up; i++) cctx[i ? 'lineTo' : 'moveTo'](tx(i), ty(D.ca[i]));
  cctx.stroke();

  // Current position dot
  if (up < L) {{
    const dx = tx(up), dy = ty(D.ca[up]);
    cctx.beginPath();
    cctx.arc(dx, dy, 4, 0, Math.PI * 2);
    cctx.fillStyle = '#2563eb';
    cctx.fill();
    cctx.strokeStyle = '#ffffff';
    cctx.lineWidth = 2;
    cctx.stroke();
  }}
}}

function up() {{
  if (step < 0) step = D.hours.length - 1;
  if (step >= D.hours.length) step = 0;
  document.getElementById('sl').max = D.hours.length - 1;
  document.getElementById('sl').value = step;
  const h = D.hours[step];
  const hh = Math.floor(h) % 24;
  const mm = Math.round((h % 1) * 60);
  document.getElementById('tm').textContent = String(hh).padStart(2, '0') + ':' + String(mm).padStart(2, '0') + '  (step ' + step + '/' + (D.hours.length - 1) + ')';

  const ca = D.ca[step], bl = D.bl[step];
  const fmt = v => v >= 1e6 ? (v / 1e6).toFixed(2) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(1) + 'K' : v.toFixed(0);
  document.getElementById('cCA').textContent = ca ? fmt(ca) + ' gCO\\u2082' : '\\u2014';
  document.getElementById('cBL').textContent = bl ? fmt(bl) + ' gCO\\u2082' : '\\u2014';
  if (ca && bl) {{
    const s = bl - ca;
    const el = document.getElementById('cSV');
    el.textContent = (s >= 0 ? '- ' : '+ ') + fmt(Math.abs(s)) + ' gCO\\u2082';
    el.style.color = s >= 0 ? 'var(--green)' : 'var(--red)';
  }}
  spawn();
  drawChart();
}}

function toggle() {{
  playing = !playing;
  const btn = document.getElementById('playBtn');
  btn.innerHTML = playing ? '&#9646;&#9646;' : '&#9654;';
  btn.classList.toggle('active', playing);
  if (playing)
    timer = setInterval(() => {{ step++; if (step >= D.hours.length) step = 0; up(); }},
      +document.getElementById('spd').value);
  else
    clearInterval(timer);
}}

function setSpd(v) {{
  if (playing) {{
    clearInterval(timer);
    timer = setInterval(() => {{ step++; if (step >= D.hours.length) step = 0; up(); }}, +v);
  }}
}}

function anim() {{ drawNet(); requestAnimationFrame(anim); }}
window.onresize = () => {{ resize(); up(); }};
resize(); up(); anim();
setTimeout(toggle, 600);
</script></body></html>"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"Animation saved: {output_path}")
    return output_path
