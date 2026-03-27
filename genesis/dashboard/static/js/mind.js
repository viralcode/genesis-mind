/**
 * Genesis Mind State — Live view of what's inside the mind
 */

const REFRESH_MS = 3000;
let refreshTimer = null;

async function fetchMindState() {
    try {
        const res = await fetch('/api/mind_state');
        if (!res.ok) return null;
        return await res.json();
    } catch { return null; }
}

function renderAll(state) {
    if (!state) return;
    renderMeta(state);
    renderNeurochemistry(state.neurochemistry || {});
    renderEmotions(state.emotions || {});
    renderDrives(state.drives || {});
    renderWorkingMemory(state.working_memory || []);
    renderConcepts(state.concepts || []);
    renderVocabulary(state.vocabulary || {});
    renderGrammar(state.grammar || {});
    renderBindings(state.cross_modal_bindings || []);
    renderEpisodes(state.episodes || []);
    renderLog(state.interaction_log || []);
}

// ═══ Meta ═══

function renderMeta(state) {
    const dev = state.development || {};
    const badge = document.getElementById('phaseBadge');
    const age = document.getElementById('ageLabel');
    const count = document.getElementById('conceptCount');

    badge.textContent = `Phase ${dev.phase || '?'}: ${dev.phase_name || 'Unknown'}`;
    age.textContent = dev.age_hours ? `${dev.age_hours}h old` : '—';
    count.textContent = `${state.concept_count || 0} concepts`;
}

// ═══ Neurochemistry ═══

function renderNeurochemistry(chem) {
    const container = document.getElementById('neurochemBars');
    const chems = ['dopamine', 'cortisol', 'serotonin', 'oxytocin'];
    let html = '';
    for (const name of chems) {
        const val = chem[name] || 0;
        const pct = Math.min(100, val * 100);
        html += `
            <div class="bar-row chem-${name}">
                <span class="bar-label">${name}</span>
                <div class="bar-track">
                    <div class="bar-fill" style="width: ${pct}%"></div>
                </div>
                <span class="bar-value">${val.toFixed(2)}</span>
            </div>`;
    }
    container.innerHTML = html;
}

// ═══ Emotions ═══

function renderEmotions(emotions) {
    const container = document.getElementById('emotionBars');
    if (!emotions || typeof emotions !== 'object' || Object.keys(emotions).length === 0) {
        container.innerHTML = '<div class="empty-state">No emotional data</div>';
        return;
    }

    const colors = {
        joy: '#fbbf24', sadness: '#60a5fa', anger: '#ef4444', fear: '#a78bfa',
        surprise: '#f472b6', disgust: '#34d399', trust: '#2dd4bf', anticipation: '#fb923c'
    };

    let html = '';
    for (const [dim, val] of Object.entries(emotions)) {
        if (typeof val !== 'number') continue;
        const pct = Math.min(100, Math.abs(val) * 100);
        const color = colors[dim] || '#94a3b8';
        html += `
            <div class="bar-row">
                <span class="bar-label">${dim}</span>
                <div class="bar-track">
                    <div class="bar-fill" style="width: ${pct}%; background: ${color}"></div>
                </div>
                <span class="bar-value">${val.toFixed(2)}</span>
            </div>`;
    }
    container.innerHTML = html || '<div class="empty-state">No emotional data</div>';
}

// ═══ Drives ═══

function renderDrives(drives) {
    const grid = document.getElementById('drivesGrid');
    if (!drives || Object.keys(drives).length === 0) {
        grid.innerHTML = '<div class="empty-state">No drive data</div>';
        return;
    }

    let html = '';
    for (const [name, info] of Object.entries(drives)) {
        const level = info.level || 0;
        const satisfied = info.satisfied;
        const cls = satisfied ? 'satisfied' : 'unsatisfied';
        const pct = Math.min(100, level * 100);
        html += `
            <div class="drive-card ${cls}">
                <div class="drive-name">${name.replace(/_/g, ' ')}</div>
                <div class="drive-bar">
                    <div class="drive-fill" style="width: ${pct}%"></div>
                </div>
            </div>`;
    }
    grid.innerHTML = html;
}

// ═══ Working Memory ═══

function renderWorkingMemory(items) {
    const container = document.getElementById('wmItems');
    if (!items || items.length === 0) {
        container.innerHTML = '<div class="empty-state">Nothing in working memory</div>';
        return;
    }

    let html = '';
    for (const item of items) {
        const sal = item.salience || 0;
        const bgAlpha = 0.1 + sal * 0.15;
        html += `
            <div class="wm-item" style="opacity: ${0.5 + sal * 0.5}">
                ${item.content || item.key}
                ${item.age_sec ? `<span style="margin-left:4px;opacity:0.4;font-size:0.7rem">${item.age_sec.toFixed(0)}s</span>` : ''}
            </div>`;
    }
    container.innerHTML = html;
}

// ═══ Concepts ═══

function renderConcepts(concepts) {
    const cloud = document.getElementById('conceptCloud');
    if (!concepts || concepts.length === 0) {
        cloud.innerHTML = '<div class="empty-state">No concepts learned yet</div>';
        return;
    }

    let html = '';
    for (const c of concepts) {
        const strength = c.strength || 0;
        let cls = 'default';
        if (c.modality === 'visual' || c.name.startsWith('proto_vision')) cls = 'visual';
        else if (c.modality === 'auditory' || c.name.startsWith('proto_sound')) cls = 'auditory';
        else if (!c.name.startsWith('proto_')) cls = 'taught';

        const dotColor = strength > 0.7 ? '#34d399' : strength > 0.4 ? '#fbbf24' : '#94a3b8';
        const size = Math.max(0.75, 0.8 + strength * 0.3);

        html += `
            <div class="concept-tag ${cls}" style="font-size: ${size}rem"
                 title="Strength: ${strength.toFixed(2)}${c.associations ? '\nLinks: ' + c.associations.join(', ') : ''}">
                <span class="strength-dot" style="background: ${dotColor}"></span>
                ${c.name}
            </div>`;
    }
    cloud.innerHTML = html;
}

// ═══ Vocabulary ═══

function renderVocabulary(vocab) {
    const panel = document.getElementById('vocabPanel');
    if (!vocab || !vocab.words || vocab.words.length === 0) {
        panel.innerHTML = '<div class="empty-state">No words learned yet</div>';
        return;
    }

    let html = '';
    for (const word of vocab.words) {
        html += `<div class="vocab-word"><span class="word">${word}</span></div>`;
    }
    html += `
        <div class="vocab-stats">
            ${vocab.total_exemplars || 0} exemplars stored ·
            ${vocab.total_recognitions || 0} recognition attempts ·
            ${((vocab.recognition_rate || 0) * 100).toFixed(0)}% accuracy
        </div>`;
    panel.innerHTML = html;
}

// ═══ Grammar ═══

function renderGrammar(grammar) {
    const panel = document.getElementById('grammarPanel');
    if (!grammar || Object.keys(grammar).length === 0) {
        panel.innerHTML = '<div class="empty-state">No grammar data</div>';
        return;
    }

    let html = '';
    html += `<div class="stat-row"><span>Mode</span><span class="val">${grammar.mode || '—'}</span></div>`;
    html += `<div class="stat-row"><span>Words heard</span><span class="val">${(grammar.words_heard || 0).toLocaleString()}</span></div>`;
    html += `<div class="stat-row"><span>Sentences</span><span class="val">${(grammar.sentences_heard || 0).toLocaleString()}</span></div>`;

    if (grammar.top_words && grammar.top_words.length > 0) {
        html += '<div style="margin-top:0.75rem;font-size:0.75rem;color:rgba(255,255,255,0.4)">Most heard:</div>';
        html += '<div class="top-words">';
        for (const tw of grammar.top_words.slice(0, 15)) {
            html += `<span class="top-word">${tw.word} (${tw.count})</span>`;
        }
        html += '</div>';
    }
    panel.innerHTML = html;
}

// ═══ Bindings ═══

function renderBindings(bindings) {
    const container = document.getElementById('bindingsList');
    if (!bindings || bindings.length === 0) {
        container.innerHTML = '<div class="empty-state">No cross-modal bindings yet</div>';
        return;
    }

    let html = '';
    for (const b of bindings) {
        html += `
            <div class="binding-card">
                <div class="binding-label">👁 ${b.visual || '?'}</div>
                <div class="binding-arrow">↕ bound ↕</div>
                <div class="binding-label">👂 ${b.auditory || '?'}</div>
                ${b.strength ? `<div style="font-size:0.7rem;color:rgba(255,255,255,0.3);margin-top:4px">strength: ${b.strength}</div>` : ''}
            </div>`;
    }
    container.innerHTML = html;
}

// ═══ Episodes ═══

function renderEpisodes(episodes) {
    const container = document.getElementById('episodesList');
    if (!episodes || episodes.length === 0) {
        container.innerHTML = '<div class="empty-state">No episodic memories stored</div>';
        return;
    }

    let html = '';
    for (const ep of episodes.slice(0, 15)) {
        const emoji = ep.emotion > 0.3 ? '😊' : ep.emotion < -0.3 ? '😢' : '📝';
        const time = ep.timestamp ? new Date(ep.timestamp).toLocaleTimeString() : '';
        html += `
            <div class="episode-card">
                <span class="episode-emoji">${emoji}</span>
                <div class="episode-body">
                    <div class="episode-summary">${ep.summary || ep.type || 'Unknown'}</div>
                    <div class="episode-time">${time} ${ep.type ? '· ' + ep.type : ''}</div>
                </div>
            </div>`;
    }
    container.innerHTML = html;
}

// ═══ Log ═══

function renderLog(log) {
    const container = document.getElementById('interactionLog');
    if (!log || log.length === 0) {
        container.innerHTML = '<div class="empty-state">No recent activity</div>';
        return;
    }

    let html = '';
    for (const entry of log.slice(0, 25)) {
        const text = typeof entry === 'string' ? entry : JSON.stringify(entry);
        html += `<div class="log-entry">${text}</div>`;
    }
    container.innerHTML = html;
}

// ═══ Main Loop ═══

async function refresh() {
    const state = await fetchMindState();
    renderAll(state);
}

// Initial load
refresh();

// Auto-refresh
refreshTimer = setInterval(refresh, REFRESH_MS);
