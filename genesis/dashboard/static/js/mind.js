/**
 * Genesis Mind Projection V7
 * Cinematographic API interface for the Neural Orb 
 */

const REFRESH_MS = 2000;
let lastKnownState = null;

// The inner monologue queue
const monologueQueue = [];
let isMonologueProcessing = false;
const seenMessages = new Set(); // To prevent spam filtering

// DOM Elements
const elOrb = document.getElementById('soul-orb');
const elGlow = document.getElementById('orb-glow');
const elStatus = document.getElementById('orb-status');
const elStream = document.getElementById('monologue-stream');
const rings = {
    sleep: document.getElementById('ring-sleep'),
    comfort: document.getElementById('ring-comfort'),
    social: document.getElementById('ring-social'),
    curiosity: document.getElementById('ring-curiosity'),
};

/**
 * 1. Fetch API state
 */
async function fetchMindState() {
    try {
        const res = await fetch('/api/mind_state');
        if (!res.ok) return null;
        return await res.json();
    } catch { return null; }
}

/**
 * 2. Main loop updates the entity
 */
function embedState(state) {
    if (!state) return;
    lastKnownState = state;

    // A) HUD Metrics (Left/Right Blocks)
    updateTelemetry(state);
    
    // B) The Deep Mind (Orb Color & Status)
    modulateOrb(state.emotions || {}, state.neurochemistry || {});
    
    // C) Drive Friction (The Rings)
    perturbRings(state.drives || {});

    // D) Inner Monologue (Queue messages from WM & Episodes)
    feedMonologue(state.working_memory || [], state.episodes || []);
    
    // E) Background Constellation (Concepts)
    drawSemanticUniverse(state.concepts || []);
}

/**
 * Update Numerical HUDs
 */
function updateTelemetry(state) {
    const dev = state.development || {};
    document.getElementById('tel-age').textContent = dev.age_hours ? dev.age_hours.toFixed(1) + 'h' : '0.0h';
    document.getElementById('tel-concepts').textContent = state.concept_count || 0;
    document.getElementById('tel-episodes').textContent = state.episode_count || 0;
    
    let vocabWords = 0;
    if (state.vocabulary && state.vocabulary.words) vocabWords = state.vocabulary.words.length;
    document.getElementById('tel-vocab').textContent = vocabWords;

    document.getElementById('phase-badge').textContent = `Phase ${dev.phase || 0}`;

    // Bar Systems
    renderSystemBars('chem-system', state.neurochemistry || {}, ['dopamine', 'serotonin', 'cortisol', 'oxytocin'], '#ffb74d');
    renderSystemBars('emotion-system', state.emotions || {}, ['arousal', 'valence', 'surprise', 'fear', 'joy'], '#81d4fa', true);
}

function renderSystemBars(containerId, data, keys, accentColor, isVector=false) {
    const el = document.getElementById(containerId);
    let html = '';
    for (const key of keys) {
        let rawVal = data[key] || 0;
        let pct = 0;
        if (isVector) {
            pct = Math.min(100, Math.abs(rawVal) * 100);
        } else {
            pct = Math.min(100, Math.max(0, rawVal) * 100);
        }
        
        let color = accentColor;
        if (isVector && rawVal < 0) color = '#ef5350'; // Negative valences are red
        if (key === 'cortisol') color = '#ef5350'; // stress is red

        html += `
            <div class="sys-bar-row">
                <div class="sys-bar-meta">
                    <span>${key}</span>
                    <span class="sys-bar-val">${rawVal.toFixed(3)}</span>
                </div>
                <div class="sys-bar-track">
                    <div class="sys-bar-fill" style="width: ${pct}%; background: ${color}; box-shadow: 0 0 5px ${color}"></div>
                </div>
            </div>
        `;
    }
    el.innerHTML = html;
}

/**
 * Central Orb Modulation (Breathing, Color Shifting)
 */
function modulateOrb(emotions, chem) {
    // Determine base HUE from Valence (-1 to 1) -> maps to (Red=0 .. Cyan=180 .. Purple=270)
    let valence = emotions.valence || 0;
    let hue = 180; // default cyan/blue
    if (valence < -0.2) hue = 350; // Red/pink 
    else if (valence > 0.3) hue = 140; // Greenish
    else if (emotions.surprise > 0.4) hue = 280; // Purple

    // Determine saturation from Dopamine
    let dopamine = chem.dopamine || 0.1;
    let sat = Math.min(100, 30 + (dopamine * 200)) + '%';
    
    // Determine brightness/pulse intensity from Arousal / Cortisol
    let arousal = Math.abs(emotions.arousal || 0);
    let scale = 1 + (arousal * 0.15); // max 15% larger
    
    // Apply dynamic CSS
    const colorA = `hsl(${hue}, ${sat}, 60%)`;
    const colorB = `hsl(${hue - 30}, ${sat}, 40%)`;
    const darkBg = `hsl(${hue}, 20%, 10%)`;
    
    elOrb.style.background = `radial-gradient(circle at 30% 30%, #ffffff, ${colorA}, ${colorB} 70%, ${darkBg})`;
    elOrb.style.boxShadow = `0 0 ${40 + arousal*40}px ${colorA}, inset -10px -10px 20px rgba(0,0,0,0.5)`;
    elOrb.style.transform = `scale(${scale})`;

    elGlow.style.background = `radial-gradient(circle, ${colorA} 0%, transparent 60%)`;
    elGlow.style.opacity = Math.min(0.8, 0.2 + (dopamine + arousal));

    // Status text
    let statusText = "Synthesizing...";
    if (chem.sleep_need > 0.8) statusText = "Reconfiguring Synapses (Tired)...";
    if (valence < -0.4) statusText = "Processing Negative Stimulus...";
    if (valence > 0.5) statusText = "Experiencing High Resonance...";
    if (emotions.surprise > 0.5) statusText = "Assimilating Novelty...";
    elStatus.textContent = statusText;
}

/**
 * Drive Tension Rings
 */
function perturbRings(drives) {
    const mapping = {
        'Sleep': rings.sleep,
        'Comfort': rings.comfort,
        'Social': rings.social,
        'Curiosity': rings.curiosity,
    };

    for (const [key, driveObj] of Object.entries(drives)) {
        const el = mapping[key];
        if (!el) continue;
        
        let level = driveObj.level || 0;
        
        // Increase border opacity and solidness as need rises
        let alpha = 0.3 + (level * 0.7);
        if (level > 0.6) el.style.borderStyle = 'solid';
        else el.style.borderStyle = 'dashed';
        el.style.borderWidth = (level > 0.5) ? '2px' : '1px';

        // Base colors
        let hue = 200;
        if (key === 'Sleep') hue = 250;
        if (key === 'Comfort') hue = 40; // yellow
        if (key === 'Social') hue = 150; // green
        if (key === 'Curiosity') hue = 280; // purple

        el.style.borderColor = `hsla(${hue}, 80%, 60%, ${alpha})`;
        el.style.boxShadow = level > 0.6 ? `0 0 15px hsla(${hue}, 80%, 60%, ${alpha * 0.5}) inset` : 'none';
    }
}

/**
 * Inner Monologue Feed
 */
function feedMonologue(wmItems, episodes) {
    // 1. Queue valid working memory concepts
    for (const item of wmItems) {
        const text = `[WM] Retaining focus on: ${item.content || item.key}`;
        if (!seenMessages.has(text)) {
            monologueQueue.push({ text, type: 'wm' });
            seenMessages.add(text);
        }
    }

    // 2. Queue recent episodes
    for (const ep of episodes) {
        const text = `[MEM] ${ep.summary || ep.type || 'Memory Snapshot Executed'}`;
        if (!seenMessages.has(text)) {
            monologueQueue.push({ text, type: 'episode' });
            seenMessages.add(text);
        }
    }

    // Keep memory cache from exploding
    if (seenMessages.size > 200) {
        const iterator = seenMessages.values();
        for (let i = 0; i < 50; i++) seenMessages.delete(iterator.next().value);
    }

    // Start consuming
    if (!isMonologueProcessing) processMonologue();
}

async function processMonologue() {
    isMonologueProcessing = true;
    while (monologueQueue.length > 0) {
        const msg = monologueQueue.shift();
        
        const div = document.createElement('div');
        div.className = `stream-item ${msg.type}`;
        div.innerHTML = `
            <span class="stream-time">${new Date().toLocaleTimeString()}</span>
            ${msg.text}
        `;
        
        elStream.appendChild(div);

        // Keep chat feed relatively short to prevent DOM bloat
        if (elStream.children.length > 15) {
            elStream.removeChild(elStream.firstChild);
        }

        // Fast staggered typing effect (150ms per thought)
        await new Promise(r => setTimeout(r, 150));
    }
    isMonologueProcessing = false;
}

/**
 * BGV: Semantic Constellation (Vis.js Network)
 * Drawn once or periodically with faint opacities. 
 */
let networkObj = null;
function drawSemanticUniverse(concepts) {
    if (concepts.length === 0) return;
    if (networkObj) return; // Only draw once per session to prevent jitter

    const container = document.getElementById('semantic-universe');
    
    // Sample up to 50 top concepts
    const nodes = [];
    const edges = [];
    
    concepts.slice(0, 50).forEach((c, i) => {
        nodes.push({ id: c.name, label: '', value: c.strength || 1 });
        
        // Form random or actual edges
        if (c.associations && Array.isArray(c.associations)) {
            c.associations.forEach(link => {
                edges.push({ from: c.name, to: link });
            });
        }
    });

    const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
    const options = {
        nodes: {
            shape: 'dot',
            color: { background: 'rgba(124, 77, 255, 0.4)', border: 'transparent' },
            font: { color: 'transparent' },
            scaling: { customScalingFunction: (v) => v, min: 2, max: 10 }
        },
        edges: {
            color: { color: 'rgba(255, 255, 255, 0.05)' },
            smooth: { type: 'continuous' }
        },
        physics: {
            barnesHut: { gravitationalConstant: -2000, centralGravity: 0.1, springLength: 200, damping: 0.05 },
            stabilization: false // Keep them floating
        },
        interaction: { dragNodes: false, zoomView: false, dragView: false }
    };

    networkObj = new vis.Network(container, data, options);
}

// ═══ Application Loop ═══
async function tick() {
    const state = await fetchMindState();
    embedState(state);
}

// Init
setInterval(tick, REFRESH_MS);
tick();
