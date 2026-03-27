// Genesis Mind V7 — Neural Explorer
// Deep introspection of all neural networks

let selectedNetwork = null;
let summaryInterval = null;
let deepInterval = null;

// ═══ Category colors ═══
const CATEGORY_COLORS = {
    sensory: { bg: 'rgba(239, 83, 80, 0.15)', accent: '#ef5350', gradient: 'linear-gradient(135deg, #ef5350, #ff7043)' },
    emotional: { bg: 'rgba(255, 213, 79, 0.15)', accent: '#ffd54f', gradient: 'linear-gradient(135deg, #ffd54f, #ffb74d)' },
    cognitive: { bg: 'rgba(79, 195, 247, 0.15)', accent: '#4fc3f7', gradient: 'linear-gradient(135deg, #4fc3f7, #81d4fa)' },
    acoustic: { bg: 'rgba(0, 230, 118, 0.15)', accent: '#00e676', gradient: 'linear-gradient(135deg, #00e676, #69f0ae)' },
};

const NETWORK_ICONS = {
    visual_cortex: '👁', limbic: '🧬', binding: '🔗', personality: '🧠',
    world_model: '🌍', meta_controller: '🎛', auditory_cortex: '👂',
    acoustic_lm: '📡', vocoder: '🗣', vq_codebook: '📊',
    phoneme_embedder: '🔤', response_decoder: '💬',
};

// ═══ Format helpers ═══
function fmtNum(n) {
    if (n === null || n === undefined) return '—';
    if (n >= 1e6) return (n / 1e6).toFixed(2) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(1) + 'K';
    return n.toLocaleString();
}

function fmtLoss(v) {
    if (v === null || v === undefined) return '—';
    if (v === 0) return '0.000';
    return v.toFixed(6);
}

// ═══ Fetch and render summary ═══
async function fetchSummary() {
    try {
        const res = await fetch('/api/explorer_summary');
        const data = await res.json();
        if (data.error) { console.error(data.error); return; }
        renderSystemBar(data.system, data.curriculum);
        renderNetworkGrid(data.networks);
    } catch (e) {
        console.error('Explorer fetch failed:', e);
    }
}

function renderSystemBar(sys, curriculum) {
    document.getElementById('sys-params').textContent = fmtNum(sys.total_params);
    document.getElementById('sys-steps').textContent = fmtNum(sys.total_training_steps);
    document.getElementById('sys-growth').textContent = sys.growth_events;
    document.getElementById('sys-phase').textContent = `Phase ${sys.phase} (${sys.phase_name})`;
    
    const badge = document.getElementById('phase-badge');
    if (badge) badge.textContent = `Phase ${sys.phase} (${sys.phase_name})`;

    // Curriculum pipeline
    const pipe = document.getElementById('curriculum-pipeline');
    if (!pipe) return;
    
    const vcLoss = curriculum.visual_cortex_loss;
    const stages = [
        { name: 'VisualCortex', detail: `loss: ${vcLoss.toFixed(4)}`, active: true },
        { name: 'Binding', detail: vcLoss < 0.05 ? 'ACTIVE' : `waiting (VC loss > 0.05)`, active: curriculum.binding_gate },
        { name: 'Personality', detail: curriculum.personality_gate ? 'ACTIVE' : 'waiting', active: curriculum.personality_gate },
        { name: 'WorldModel', detail: curriculum.world_model_gate ? 'ACTIVE' : 'waiting', active: curriculum.world_model_gate },
    ];
    
    pipe.innerHTML = stages.map((s, i) => {
        const cls = s.active ? 'active' : 'gated';
        const arrow = i < stages.length - 1 ? '<span class="pipe-arrow">→</span>' : '';
        return `<span class="pipe-node ${cls}" title="${s.detail}">${s.name}</span>${arrow}`;
    }).join('');
}

function renderNetworkGrid(networks) {
    const grid = document.getElementById('network-grid');
    if (!grid) return;
    
    // Sort: cognitive first, then sensory, emotional, acoustic
    const order = ['sensory', 'emotional', 'cognitive', 'acoustic'];
    const sorted = Object.entries(networks).sort((a, b) => {
        const oa = order.indexOf(a[1].category || 'cognitive');
        const ob = order.indexOf(b[1].category || 'cognitive');
        return oa - ob;
    });

    // Only rebuild if card count changed
    if (grid.children.length !== sorted.length) {
        grid.innerHTML = sorted.map(([name, net]) => buildCardHTML(name, net)).join('');
        // Attach click handlers
        grid.querySelectorAll('.net-card').forEach(card => {
            card.addEventListener('click', () => selectNetwork(card.dataset.name));
        });
    } else {
        // Update in-place
        sorted.forEach(([name, net]) => {
            updateCardInPlace(name, net);
        });
    }
}

function buildCardHTML(name, net) {
    const icon = NETWORK_ICONS[name] || '🧩';
    const displayName = name.replace(/_/g, ' ');
    const cat = net.category || 'cognitive';
    const sel = selectedNetwork === name ? 'selected' : '';
    
    return `
    <div class="net-card ${sel}" data-name="${name}" data-category="${cat}" id="card-${name}">
        <div class="net-card-header">
            <span class="net-card-name">${icon} ${displayName}</span>
            <span class="health-dot ${net.health || 'idle'}" title="${net.health || 'idle'}"></span>
        </div>
        <div class="net-card-desc">${net.description || ''}</div>
        <div class="net-card-stats">
            <div class="net-stat">
                <span class="net-stat-label">Params</span>
                <span class="net-stat-value" id="stat-params-${name}">${fmtNum(net.params)}</span>
            </div>
            <div class="net-stat">
                <span class="net-stat-label">Train Steps</span>
                <span class="net-stat-value" id="stat-steps-${name}">${fmtNum(net.training_steps)}</span>
            </div>
            <div class="net-stat">
                <span class="net-stat-label">Avg Loss</span>
                <span class="net-stat-value loss" id="stat-loss-${name}">${fmtLoss(net.avg_loss)}</span>
            </div>
            <div class="net-stat">
                <span class="net-stat-label">Status</span>
                <span class="net-stat-value" id="stat-health-${name}" style="text-transform:uppercase; font-size:0.65rem;">${net.health || 'idle'}</span>
            </div>
        </div>
    </div>`;
}

function updateCardInPlace(name, net) {
    const el = (id) => document.getElementById(id);
    const p = el(`stat-params-${name}`);
    if (p) p.textContent = fmtNum(net.params);
    const s = el(`stat-steps-${name}`);
    if (s) s.textContent = fmtNum(net.training_steps);
    const l = el(`stat-loss-${name}`);
    if (l) l.textContent = fmtLoss(net.avg_loss);
    const h = el(`stat-health-${name}`);
    if (h) h.textContent = net.health || 'idle';
    
    // Update health dot
    const card = el(`card-${name}`);
    if (card) {
        const dot = card.querySelector('.health-dot');
        if (dot) { dot.className = `health-dot ${net.health || 'idle'}`; }
        card.classList.toggle('selected', selectedNetwork === name);
    }
}

// ═══ Network selection → deep dive ═══
function selectNetwork(name) {
    if (selectedNetwork === name) {
        // Deselect
        selectedNetwork = null;
        document.getElementById('deep-dive').style.display = 'none';
        if (deepInterval) { clearInterval(deepInterval); deepInterval = null; }
        document.querySelectorAll('.net-card').forEach(c => c.classList.remove('selected'));
        return;
    }
    
    selectedNetwork = name;
    document.querySelectorAll('.net-card').forEach(c => {
        c.classList.toggle('selected', c.dataset.name === name);
    });
    
    document.getElementById('deep-dive').style.display = '';
    fetchDeepDive(name);
    
    if (deepInterval) clearInterval(deepInterval);
    deepInterval = setInterval(() => fetchDeepDive(name), 5000);
    
    // Scroll to deep dive
    setTimeout(() => {
        document.getElementById('deep-dive').scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
}

async function fetchDeepDive(name) {
    try {
        const res = await fetch(`/api/network_deep/${name}`);
        const data = await res.json();
        if (data.error) { console.error(data.error); return; }
        renderDeepDive(data);
    } catch (e) {
        console.error('Deep dive fetch failed:', e);
    }
}

function renderDeepDive(data) {
    const icon = NETWORK_ICONS[data.name] || '🧩';
    const displayName = data.name.replace(/_/g, ' ');
    document.getElementById('dd-title').textContent = `${icon} ${displayName}`;
    document.getElementById('dd-desc').textContent = data.description || '';
    
    renderArchitecture(data.layers, data.decoder_layers);
    renderStats(data.stats || {});
    renderHistograms(data.weight_stats || {}, data.category);
    renderActivations(data);
}

function renderArchitecture(layers, decoderLayers) {
    const container = document.getElementById('dd-layers');
    if (!layers || layers.length === 0) {
        container.innerHTML = '<div style="color:var(--text-muted);font-size:0.75rem;">No layers available (module may be a callable)</div>';
        return;
    }
    
    let html = '';
    
    // Group by module prefix
    const groups = {};
    layers.forEach(l => {
        const parts = l.name.split('.');
        const group = parts.length > 1 ? parts.slice(0, -1).join('.') : 'root';
        if (!groups[group]) groups[group] = [];
        groups[group].push(l);
    });
    
    for (const [group, params] of Object.entries(groups)) {
        const totalParams = params.reduce((s, p) => s + p.params, 0);
        html += `<div class="arch-layer" style="border-left-color: var(--accent); background: rgba(79,195,247,0.05);">
            <span class="arch-layer-name" title="${group}">${group}</span>
            <span class="arch-layer-shape">${fmtNum(totalParams)}p</span>
        </div>`;
        params.forEach(p => {
            const shape = p.shape.join('×');
            html += `<div class="arch-layer" style="padding-left: 1.2rem; border-left-color: rgba(79,195,247,0.15);">
                <span class="arch-layer-name" title="${p.name}">${p.name.split('.').pop()}</span>
                <span class="arch-layer-shape">[${shape}]</span>
            </div>`;
        });
    }
    
    // Decoder layers (visual_cortex)
    if (decoderLayers && decoderLayers.length > 0) {
        html += `<div class="arch-layer" style="border-left-color: #ff7043; background: rgba(239,83,80,0.05); margin-top: 0.5rem;">
            <span class="arch-layer-name">decoder</span>
            <span class="arch-layer-shape">${fmtNum(decoderLayers.reduce((s,l) => s + l.params, 0))}p</span>
        </div>`;
    }
    
    container.innerHTML = html;
}

function renderStats(stats) {
    const container = document.getElementById('dd-stats');
    if (!stats || Object.keys(stats).length === 0) {
        container.innerHTML = '';
        return;
    }
    
    // Map common stat keys to display labels
    const labels = {
        total_experiences: 'Experiences', experiences: 'Experiences',
        training_steps: 'Train Steps', train_steps: 'Train Steps',
        avg_loss: 'Avg Loss', has_consciousness: 'Consciousness',
        params: 'Parameters', total_params: 'Parameters',
        buffer_size: 'Buffer Size', frames_seen: 'Frames Seen',
        reactions: 'Reactions', bindings_created: 'Bindings',
        predictions_made: 'Predictions', total_routes: 'Routes',
        dominant_module: 'Dominant', dominant_weight: 'Dom. Weight',
        temperature: 'Temperature',
    };
    
    let html = '';
    for (const [key, val] of Object.entries(stats)) {
        const label = labels[key] || key.replace(/_/g, ' ');
        let display = val;
        if (typeof val === 'number') {
            display = key.includes('loss') || key.includes('surprise') || key.includes('temperature') 
                ? fmtLoss(val) 
                : fmtNum(val);
        } else if (typeof val === 'boolean') {
            display = val ? '🟢 Yes' : '⚫ No';
        } else if (typeof val === 'object') {
            continue; // Skip complex objects
        }
        
        html += `<div class="dd-stat-item">
            <span class="net-stat-label">${label}</span>
            <span class="net-stat-value">${display}</span>
        </div>`;
    }
    container.innerHTML = html;
}

function renderHistograms(weightStats, category) {
    const container = document.getElementById('dd-histograms');
    if (!weightStats || Object.keys(weightStats).length === 0) {
        container.innerHTML = '<div style="color:var(--text-muted);font-size:0.75rem;padding:1rem;">No weight data available</div>';
        return;
    }
    
    const colors = CATEGORY_COLORS[category] || CATEGORY_COLORS.cognitive;
    
    let html = '';
    for (const [name, ws] of Object.entries(weightStats)) {
        const canvasId = `hist-${name.replace(/\./g, '-')}`;
        html += `<div class="hist-card">
            <div class="hist-title" title="${name}">${name}</div>
            <canvas class="hist-canvas" id="${canvasId}" width="300" height="50"></canvas>
            <div class="hist-meta">
                <span>μ=${ws.mean.toFixed(4)}</span>
                <span>σ=${ws.std.toFixed(4)}</span>
                <span>‖w‖=${ws.norm.toFixed(2)}</span>
                ${ws.grad_norm !== undefined ? `<span>∇=${ws.grad_norm.toFixed(4)}</span>` : ''}
            </div>
        </div>`;
    }
    container.innerHTML = html;
    
    // Draw histograms on canvas
    requestAnimationFrame(() => {
        for (const [name, ws] of Object.entries(weightStats)) {
            if (!ws.histogram) continue;
            const canvasId = `hist-${name.replace(/\./g, '-')}`;
            drawHistogram(canvasId, ws.histogram, colors.accent);
        }
    });
}

function drawHistogram(canvasId, hist, color) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const { counts, edges } = hist;
    
    ctx.clearRect(0, 0, w, h);
    
    const maxCount = Math.max(...counts, 1);
    const barWidth = w / counts.length;
    
    // Parse hex color for rgba
    const hexToRgb = (hex) => {
        const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
        return result ? {
            r: parseInt(result[1], 16), g: parseInt(result[2], 16), b: parseInt(result[3], 16)
        } : { r: 79, g: 195, b: 247 };
    };
    const rgb = hexToRgb(color);
    
    counts.forEach((count, i) => {
        const barHeight = (count / maxCount) * (h - 4);
        const x = i * barWidth;
        const y = h - barHeight - 2;
        
        // Gradient fill
        const alpha = 0.3 + (count / maxCount) * 0.5;
        ctx.fillStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
        ctx.fillRect(x + 0.5, y, barWidth - 1, barHeight);
        
        // Top edge highlight
        ctx.fillStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, 0.8)`;
        ctx.fillRect(x + 0.5, y, barWidth - 1, 1);
    });
    
    // Zero line
    const zeroIdx = edges.findIndex(e => e >= 0);
    if (zeroIdx > 0 && zeroIdx < edges.length) {
        const zeroX = (zeroIdx / edges.length) * w;
        ctx.strokeStyle = 'rgba(255,255,255,0.2)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(zeroX, 0);
        ctx.lineTo(zeroX, h);
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

function renderActivations(data) {
    const container = document.getElementById('dd-activation-content');
    const activations = data.activations || {};
    
    if (Object.keys(activations).length === 0) {
        container.innerHTML = '<div style="color:var(--text-muted);font-size:0.75rem;">No live activation data</div>';
        return;
    }
    
    let html = '';
    
    // Personality hidden state heatmap
    if (activations.hidden_state) {
        html += '<div style="margin-bottom:1rem;">';
        html += '<div style="font-size:0.65rem;color:var(--text-muted);margin-bottom:0.4rem;">HIDDEN STATE (128-dim)</div>';
        html += '<div class="activation-heatmap">';
        activations.hidden_state.forEach(v => {
            const intensity = Math.min(Math.abs(v) * 5, 1);
            const hue = v >= 0 ? 190 : 10; // Blue for positive, red for negative
            const bg = `hsla(${hue}, 80%, 60%, ${intensity})`;
            html += `<div class="activation-cell" style="background:${bg};" title="${v.toFixed(4)}"></div>`;
        });
        html += '</div>';
        html += `<div style="font-size:0.55rem;color:var(--text-muted);margin-top:0.3rem;">Buffer: ${activations.experience_buffer_size || 0} experiences</div>`;
        html += '</div>';
    }
    
    // Meta-controller routing
    if (activations.routing) {
        html += '<div style="margin-bottom:1rem;">';
        html += '<div style="font-size:0.65rem;color:var(--text-muted);margin-bottom:0.4rem;">ROUTING WEIGHTS</div>';
        for (const [name, weight] of Object.entries(activations.routing)) {
            const pct = (weight * 100).toFixed(1);
            html += `<div class="routing-bar">
                <span class="routing-bar-label">${name}</span>
                <div class="routing-bar-track"><div class="routing-bar-fill" style="width:${pct}%"></div></div>
                <span class="routing-bar-value">${pct}%</span>
            </div>`;
        }
        html += '</div>';
    }
    
    // VQ Codebook usage
    if (activations.codebook_usage) {
        html += '<div style="margin-bottom:1rem;">';
        html += `<div style="font-size:0.65rem;color:var(--text-muted);margin-bottom:0.4rem;">CODEBOOK USAGE (${activations.active_codes}/${activations.total_codes} active)</div>`;
        html += '<div class="codebook-grid">';
        activations.codebook_usage.forEach(usage => {
            const intensity = Math.min(usage / 50, 1); // Normalize to max ~50 uses
            const bg = `rgba(0, 230, 118, ${0.05 + intensity * 0.8})`;
            html += `<div class="codebook-cell" style="background:${bg};" title="Uses: ${usage}"></div>`;
        });
        html += '</div>';
        html += '</div>';
    }
    
    // Visual cortex specific
    if (activations.frames_seen !== undefined) {
        html += `<div style="margin-bottom:0.5rem;">
            <div style="font-size:0.65rem;color:var(--text-muted);">FRAMES PROCESSED</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;color:#ef5350;">${fmtNum(activations.frames_seen)}</div>
        </div>`;
        html += `<div>
            <div style="font-size:0.65rem;color:var(--text-muted);">ENCODER TRAIN STEPS</div>
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.2rem;color:#ef5350;">${fmtNum(activations.train_steps)}</div>
        </div>`;
    }
    
    // Generic fallback for any other activation data
    const rendered = new Set(['hidden_state', 'experience_buffer_size', 'routing', 'codebook_usage', 'active_codes', 'total_codes', 'frames_seen', 'train_steps']);
    const remaining = Object.entries(activations).filter(([k]) => !rendered.has(k));
    if (remaining.length > 0) {
        remaining.forEach(([key, val]) => {
            const display = typeof val === 'number' ? fmtNum(val) : JSON.stringify(val);
            html += `<div style="margin-bottom:0.4rem;">
                <span style="font-size:0.6rem;color:var(--text-muted);text-transform:uppercase;">${key.replace(/_/g,' ')}</span>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.8rem;">${display}</div>
            </div>`;
        });
    }
    
    container.innerHTML = html;
}

// ═══ Close handler ═══
document.getElementById('dd-close')?.addEventListener('click', () => {
    selectedNetwork = null;
    document.getElementById('deep-dive').style.display = 'none';
    if (deepInterval) { clearInterval(deepInterval); deepInterval = null; }
    document.querySelectorAll('.net-card').forEach(c => c.classList.remove('selected'));
});

// ═══ Init ═══
fetchSummary();
summaryInterval = setInterval(fetchSummary, 3000);
