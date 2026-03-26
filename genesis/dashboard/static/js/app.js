// Genesis Mind V5 — Dashboard Application Logic

let emotionChart = null;
let drivesChart = null;

// UI Elements
const phaseBadge = document.getElementById('phase-badge');
const threadsContainer = document.getElementById('threads-container');
const sensesContainer = document.getElementById('senses-container');
const wmContainer = document.getElementById('wm-container');
const wmUsage = document.getElementById('wm-usage');
const wmCapacity = document.getElementById('wm-capacity');
const hiddenStateContainer = document.getElementById('hidden-state-container');
const activityStream = document.getElementById('activity-stream');
const nodeCount = document.getElementById('node-count');

// Vis Network configuration
let network = null;
let nodesData = new vis.DataSet([]);
let edgesData = new vis.DataSet([]);
const networkContainer = document.getElementById('network-canvas');

const brainOptions = {
    nodes: {
        shape: 'dot',
        borderWidth: 0,
        color: { background: 'rgba(79, 195, 247, 0.2)' }
    },
    edges: {
        width: 1,
        color: 'rgba(255,255,255,0.08)',
        smooth: false
    },
    groups: {
        input: { size: 14, color: {background: 'rgba(239, 83, 80, 0.7)'} },
        hidden: { size: 8, color: {background: 'rgba(79, 195, 247, 0.1)'} },
        output: { size: 14, color: {background: 'rgba(255, 213, 79, 0.7)'} }
    },
    physics: false, // Freeze physics to keep architecture rigid
    interaction: { zoomView: true, dragView: true }
};

function initNetwork() {
    if (!networkContainer) return;
    
    // Build the 144-node Neural Architecture
    let nodes = [];
    let edges = [];
    
    // 1. Sensory Input Layer (8 nodes)
    for (let i = 0; i < 8; i++) {
        nodes.push({id: `in_${i}`, label: '', group: 'input', x: -400, y: (i * 50) - 175, title: `Sensory/Limbic Node ${i}`});
    }
    
    // 2. Hidden State (128 nodes, 16x8 grid)
    let hIdx = 0;
    for (let col = 0; col < 16; col++) {
        for (let row = 0; row < 8; row++) {
            nodes.push({
                id: `h_${hIdx}`, 
                label: '', 
                group: 'hidden', 
                x: -200 + (col * 25), 
                y: (row * 30) - 105,
                title: `Hidden Activation (GRU Unit ${hIdx})`
            });
            hIdx++;
        }
    }
    
    // 3. World Model / Output (8 nodes)
    for (let i = 0; i < 8; i++) {
        nodes.push({id: `out_${i}`, label: '', group: 'output', x: 250, y: (i * 50) - 175, title: `World Model Prediction Node ${i}`});
    }

    // Wiring Input -> Hidden
    for (let i = 0; i < 8; i++) {
        for (let col = 0; col < 2; col++) {
            for (let row = 0; row < 8; row++) {
                if (Math.random() > 0.6) edges.push({from: `in_${i}`, to: `h_${col*8 + row}`});
            }
        }
    }
    // Wiring Hidden -> Hidden (Dense-like)
    for (let col = 0; col < 15; col++) {
        for (let row = 0; row < 8; row++) {
            if (Math.random() > 0.4) {
                // connect to next column
                edges.push({from: `h_${col*8 + row}`, to: `h_${(col+1)*8 + Math.floor(Math.random()*8)}`});
            }
        }
    }
    // Wiring Hidden -> Output
    for (let i = 0; i < 8; i++) {
        for (let col = 14; col < 16; col++) {
            for (let row = 0; row < 8; row++) {
                if (Math.random() > 0.6) edges.push({from: `h_${col*8 + row}`, to: `out_${i}`});
            }
        }
    }

    nodesData = new vis.DataSet(nodes);
    edgesData = new vis.DataSet(edges);
    
    network = new vis.Network(networkContainer, {nodes: nodesData, edges: edgesData}, brainOptions);
    
    // Center it nicely after spawn
    setTimeout(() => network.fit({animation: {duration: 1000, easingFunction: 'easeInOutQuad'}}), 500);
}

// Stats Elements
const conceptCount = document.getElementById('concept-count');
const expCount = document.getElementById('exp-count');
const surpriseLoss = document.getElementById('surprise-loss');

// Chemistry Bars
const chemDopamine = document.getElementById('chem-dopamine');
const chemCortisol = document.getElementById('chem-cortisol');
const chemSerotonin = document.getElementById('chem-serotonin');
const chemOxytocin = document.getElementById('chem-oxytocin');
const valDopamine = document.getElementById('val-dopamine');
const valCortisol = document.getElementById('val-cortisol');
const valSerotonin = document.getElementById('val-serotonin');
const valOxytocin = document.getElementById('val-oxytocin');

// Pre-create 128 hidden state cells
function initHiddenStateViz() {
    if (!hiddenStateContainer) return;
    for (let i = 0; i < 128; i++) {
        const cell = document.createElement('div');
        cell.className = 'hs-cell';
        hiddenStateContainer.appendChild(cell);
    }
}

// Initialize Chart.js configuration
function initCharts() {
    Chart.defaults.color = '#8a8a93';
    Chart.defaults.font.family = 'Inter';

    // 8D Emotional State Radar Chart
    const elEmotion = document.getElementById('emotionChart');
    if (elEmotion) {
        const ctxEmotion = elEmotion.getContext('2d');
        emotionChart = new Chart(ctxEmotion, {
            type: 'radar',
            data: {
                labels: ['Joy', 'Fear', 'Anger', 'Sadness', 'Trust', 'Disgust', 'Anticipation', 'Surprise'],
                datasets: [{
                    label: 'Current Emotional Vector',
                    data: [0, 0, 0, 0, 0, 0, 0, 0],
                    backgroundColor: 'rgba(79, 195, 247, 0.2)',
                    borderColor: '#4fc3f7',
                    pointBackgroundColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#4fc3f7',
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: '#f0f0f5', font: { size: 11 } },
                        ticks: { display: false, max: 1.0, min: -0.2 }
                    }
                },
                plugins: { legend: { display: false } }
            }
        });
    }

    // 8 Maslow Drives Bar Chart
    const elDrives = document.getElementById('drivesChart');
    if (elDrives) {
        const ctxDrives = elDrives.getContext('2d');
        drivesChart = new Chart(ctxDrives, {
            type: 'bar',
            data: {
                labels: ['Sleep', 'Comfort', 'Social', 'Belonging', 'Curiosity', 'Novelty', 'Mastery', 'Autonomy'],
                datasets: [{
                    label: 'Drive Activation',
                    data: [0,0,0,0,0,0,0,0],
                    backgroundColor: [
                        '#ef5350', '#ef5350', // Survival
                        '#ffd54f', '#ffd54f', // Social (Tier 2)
                        '#4fc3f7', '#4fc3f7', '#4fc3f7', // Cognitive (Tier 3)
                        '#ba68c8'             // Self (Tier 4)
                    ],
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        max: 1.0,
                        min: 0,
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    },
                    y: {
                        grid: { display: false }
                    }
                },
                plugins: { legend: { display: false } }
            }
        });
    }
}

function updateUI(state) {
    if (state.status === "booting...") return;

    // Phase Status
    if (phaseBadge) phaseBadge.textContent = `Phase ${state.core.phase} (${state.core.phase_name})`;

    // Threads
    if (state.threads && threadsContainer) {
        threadsContainer.innerHTML = '';
        for (const [name, data] of Object.entries(state.threads)) {
            threadsContainer.innerHTML += `
                <div class="thread-card">
                    <span class="thread-name">${name}</span>
                    <span class="thread-ticks">${data.ticks} ticks | ${data.errors} errors</span>
                </div>
            `;
        }
    }

    // Senses
    if (state.senses && sensesContainer) {
        sensesContainer.innerHTML = `
            <div class="thread-card">
                <span class="thread-name">Vision (Camera)</span>
                <span class="thread-ticks" style="color:var(--accent)">${state.senses.vision}</span>
            </div>
            <div class="thread-card">
                <span class="thread-name">Auditory (Mic)</span>
                <span class="thread-ticks" style="color:var(--accent)">${state.senses.auditory}</span>
            </div>
            <div class="thread-card" style="grid-column: 1 / -1;">
                <span class="thread-name">Proprioception (Body)</span>
                <span class="thread-ticks">Time: ${state.senses.proprioception.time_of_day} | Fatigue: ${state.senses.proprioception.fatigue.toFixed(2)} | Uptime: ${state.senses.proprioception.uptime_hours.toFixed(1)}h</span>
            </div>
        `;
    }

    // Activity Stream
    if (state.stream && activityStream) {
        // Only autoscroll if user is already at the bottom
        let isScrolledToBottom = activityStream.scrollHeight - activityStream.clientHeight <= activityStream.scrollTop + 5;
        
        // Prevent unnecessary DOM writes if nothing changed (basic length check for efficiency)
        if (activityStream.children.length !== state.stream.length || state.stream.length > 0) {
            let html = '';
            state.stream.forEach(item => {
                const sourceClass = item.prefix === '💭' ? 'thought' : item.prefix === '👂' ? 'heard' : 'dream';
                const label = item.prefix === '💭' ? 'THOUGHT' : item.prefix === '👂' ? 'PERCEPTION' : 'SYSTEM';
                html += `
                    <div class="stream-item ${sourceClass}">
                        <div class="stream-time">
                            <span class="stream-source">${label}</span>
                            <span>${item.time}</span>
                        </div>
                        <div>${item.prefix} ${item.message}</div>
                    </div>
                `;
            });
            activityStream.innerHTML = html;
            if (isScrolledToBottom) {
                activityStream.scrollTop = activityStream.scrollHeight;
            }
        }
    }

    // Network Graph Real-Time Neural Activations update
    if (network && state.neural && state.neural.layer3_personality) {
        if (nodeCount) nodeCount.textContent = 144; // Total nodes simulated
        
        const hs = state.neural.layer3_personality.hidden_state_activation || [];
        const loss = state.neural.layer4_world_model.last_loss || 0;
        let updates = [];
        
        // Flash hidden layer neurons based on live model hidden state
        for (let i = 0; i < hs.length && i < 128; i++) {
            const val = hs[i];
            const intensity = Math.min(Math.max((val + 1) / 2, 0.1), 1);
            updates.push({
                id: `h_${i}`,
                color: { background: `rgba(79, 195, 247, ${intensity})` },
                size: 6 + (intensity * 6) // pulse size slightly
            });
        }
        
        // Pulse Sensory and World Model randomly or based on systemic noise
        const inputNoise = Math.random() * 0.4 + 0.3;
        for (let i=0; i<8; i++) {
            updates.push({id: `in_${i}`, size: 12 + (Math.random() * 6), color: {background: `rgba(239, 83, 80, ${inputNoise + (Math.random()*0.3)})`}});
            
            // Output node flashes based on World Model prediction loss 
            const outIntensity = Math.min(0.2 + (loss * 10), 0.9) + (Math.random()*0.2);
            updates.push({id: `out_${i}`, size: 12 + (Math.random() * 4), color: {background: `rgba(255, 213, 79, ${outIntensity})`}});
        }
        
        if (nodesData.get('h_0')) {
             nodesData.update(updates);
        }
    }

    // Working Memory
    if (state.working_memory && wmUsage && wmCapacity && wmContainer) {
        wmUsage.textContent = state.working_memory.usage;
        wmCapacity.textContent = state.working_memory.capacity;
        wmContainer.innerHTML = '';
        
        if (state.working_memory.slots.length === 0) {
            wmContainer.innerHTML = '<div style="color:var(--text-muted); font-size:0.8rem; padding:1rem; text-align:center;">Buffer is empty</div>';
        } else {
            state.working_memory.slots.forEach(slot => {
                wmContainer.innerHTML += `
                    <div class="wm-slot">
                        <span class="wm-concept">${slot.concept}</span>
                        <div class="wm-meta">
                            <span>${slot.source}</span>
                            <span class="wm-salience">${slot.salience.toFixed(2)}</span>
                        </div>
                    </div>
                `;
            });
        }
    }

    // Neurochemistry
    if (state.neurochemistry && chemDopamine && chemCortisol && chemSerotonin && chemOxytocin) {
        const d = state.neurochemistry.dopamine || 0;
        const c = state.neurochemistry.cortisol || 0;
        const s = state.neurochemistry.serotonin || 0;
        const o = state.neurochemistry.oxytocin || 0;
        
        chemDopamine.style.width = `${d * 100}%`;
        chemCortisol.style.width = `${c * 100}%`;
        chemSerotonin.style.width = `${s * 100}%`;
        chemOxytocin.style.width = `${o * 100}%`;
        
        if (valDopamine) valDopamine.textContent = d.toFixed(3);
        if (valCortisol) valCortisol.textContent = c.toFixed(3);
        if (valSerotonin) valSerotonin.textContent = s.toFixed(3);
        if (valOxytocin) valOxytocin.textContent = o.toFixed(3);
    }

    // Neural Stats
    if (state.neural) {
        if (conceptCount) conceptCount.textContent = state.neural.layer2_binding.learned_concepts.toLocaleString();
        if (expCount) expCount.textContent = state.neural.layer3_personality.total_experiences.toLocaleString();
        if (surpriseLoss) surpriseLoss.textContent = state.neural.layer4_world_model.last_loss.toFixed(4);

        // Update hidden state cells
        if (hiddenStateContainer) {
            const cells = hiddenStateContainer.children;
            const hs = state.neural.layer3_personality.hidden_state_activation || [];
            for (let i = 0; i < cells.length; i++) {
                if (i < hs.length) {
                    // Map activation value to opacity/color
                    const val = hs[i];
                    // normalize roughly from -1 to 1 for visual
                    const intensity = Math.min(Math.max((val + 1) / 2, 0), 1);
                    cells[i].style.backgroundColor = `rgba(79, 195, 247, ${intensity})`;
                } else {
                    cells[i].style.backgroundColor = 'rgba(255,255,255,0.05)';
                }
            }
        }
    }

    // Emotions Radar Chart Update
    if (state.emotions && emotionChart) {
        emotionChart.data.datasets[0].data = state.emotions;
        emotionChart.update('none'); // Update without full animation for performance
    }

    // Drives Bar Chart Update
    if (state.drives && drivesChart) {
        drivesChart.data.datasets[0].data = [
            state.drives.sleep?.level || 0,
            state.drives.comfort?.level || 0,
            state.drives.social?.level || 0,
            state.drives.belonging?.level || 0,
            state.drives.curiosity?.level || 0,
            state.drives.novelty?.level || 0,
            state.drives.mastery?.level || 0,
            state.drives.autonomy?.level || 0,
        ];
        drivesChart.update('none');
    }
}

// Fetch loop
async function fetchState() {
    try {
        const response = await fetch('/api/state');
        if (response.ok) {
            const data = await response.json();
            updateUI(data);
        }
    } catch (e) {
        console.error("Dashboard disconnected", e);
    }
}

// Init
window.addEventListener('DOMContentLoaded', () => {
    initHiddenStateViz();
    initCharts();
    initNetwork();
    
    // Poll every 1 second
    setInterval(fetchState, 1000);
    fetchState(); // initial fetch
});
