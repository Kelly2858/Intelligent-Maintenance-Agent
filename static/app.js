/* =========================================
   Intelligent Maintenance Agent — App Logic
   ========================================= */

const API = '';
let lastResult = null;

// ---- Submit complaint ----
async function submitComplaint() {
    const input = document.getElementById('complaintInput');
    const btn = document.getElementById('submitBtn');
    const text = input.value.trim();
    if (!text) { input.focus(); return; }

    btn.disabled = true;
    btn.querySelector('.btn-text').style.display = 'none';
    btn.querySelector('.btn-loader').style.display = 'inline';

    try {
        const res = await fetch(`${API}/api/complaint`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ complaint: text }),
        });
        if (!res.ok) {
            const err = await res.json();
            alert('Error: ' + (err.detail || 'Request failed'));
            return;
        }
        lastResult = await res.json();
        displayResult(lastResult);
        loadTickets();
        loadStats();
    } catch (e) {
        alert('Network error: ' + e.message);
    } finally {
        btn.disabled = false;
        btn.querySelector('.btn-text').style.display = 'inline';
        btn.querySelector('.btn-loader').style.display = 'none';
    }
}

// ---- Display result ----
function displayResult(data) {
    const section = document.getElementById('resultSection');
    section.classList.remove('hidden');

    document.getElementById('ticketId').textContent = `#${data.ticket_id}`;
    document.getElementById('resCategory').textContent = data.classification.category;
    document.getElementById('resConfidence').textContent = `Confidence: ${data.classification.confidence_pct}`;
    document.getElementById('resPriority').textContent = data.priority_assessment.priority;
    document.getElementById('resPriorityReason').textContent = data.priority_assessment.reasoning;

    // Category box styling
    const catBox = document.getElementById('categoryBox');
    catBox.className = `result-box category-box cat-${data.classification.category}`;

    // Priority box styling
    const priBox = document.getElementById('priorityBox');
    priBox.className = `result-box priority-box pri-${data.priority_assessment.priority}`;

    // Keywords
    const kwContainer = document.getElementById('resKeywords');
    kwContainer.innerHTML = data.analysis.keywords_matched.length
        ? data.analysis.keywords_matched.map(k => `<span class="kw-tag">${k}</span>`).join('')
        : '<span style="color:var(--text-muted);font-size:0.8rem">No keywords matched</span>';

    // Score bars
    const scoresContainer = document.getElementById('resScores');
    scoresContainer.innerHTML = '';
    for (const [cat, score] of Object.entries(data.classification.scores)) {
        const pct = Math.round(score * 100);
        scoresContainer.innerHTML += `
            <div class="score-row">
                <span class="score-label">${cat}</span>
                <div class="score-track">
                    <div class="score-fill ${cat}" style="width:${pct}%"></div>
                </div>
                <span class="score-pct">${pct}%</span>
            </div>`;
    }

    // JSON
    document.getElementById('jsonOutput').textContent = JSON.stringify(data, null, 2);

    // Scroll to result
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---- Copy JSON ----
function copyJSON() {
    if (!lastResult) return;
    navigator.clipboard.writeText(JSON.stringify(lastResult, null, 2));
    const btn = document.querySelector('.copy-btn');
    btn.textContent = '✅ Copied!';
    setTimeout(() => btn.textContent = '📋 Copy', 1500);
}

// ---- Use sample ----
function useSample(el) {
    document.getElementById('complaintInput').value = el.textContent;
    document.getElementById('complaintInput').focus();
}

// ---- Load tickets ----
async function loadTickets() {
    const cat = document.getElementById('filterCategory').value;
    const pri = document.getElementById('filterPriority').value;
    let url = `${API}/api/tickets?limit=30`;
    if (cat) url += `&category=${cat}`;
    if (pri) url += `&priority=${pri}`;

    try {
        const res = await fetch(url);
        const tickets = await res.json();
        const container = document.getElementById('ticketsList');

        if (!tickets.length) {
            container.innerHTML = '<p class="empty-msg">No tickets found.</p>';
            return;
        }

        container.innerHTML = tickets.map(t => `
            <div class="ticket-row" style="display:flex; align-items:center; gap:10px;">
                <input type="checkbox" class="ticket-checkbox" value="${t.id}" onchange="toggleDeleteBtn()" style="cursor:pointer; width:16px; height:16px; accent-color:var(--pri-high);">
                <span class="t-id" style="min-width:40px;">#${t.id}</span>
                <span class="t-complaint" style="flex:1;" title="${t.complaint.replace(/"/g, '&quot;')}">${t.complaint}</span>
                <span class="t-badge cat-${t.category}">${t.category}</span>
                <span class="t-badge pri-${t.priority}">${t.priority}</span>
            </div>
        `).join('');
        toggleDeleteBtn();
    } catch (e) {
        console.error('Failed to load tickets:', e);
    }
}

// ---- Load stats ----
async function loadStats() {
    try {
        const res = await fetch(`${API}/api/stats`);
        const stats = await res.json();
        document.querySelector('#statTotal .stat-num').textContent = stats.total_tickets;
        document.querySelector('#statHigh .stat-num').textContent = stats.by_priority.High || 0;
        document.querySelector('#statMed .stat-num').textContent = stats.by_priority.Medium || 0;
        document.querySelector('#statLow .stat-num').textContent = stats.by_priority.Low || 0;
    } catch (e) {
        console.error('Failed to load stats:', e);
    }
}

// ---- Train AI Model ----
async function trainAI() {
    const btn = document.getElementById('trainBtn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<span>⏳ Training...</span>';
    btn.disabled = true;

    try {
        const res = await fetch(`${API}/api/train`, { method: 'POST' });
        const data = await res.json();
        
        if (res.ok) {
            alert(`Success: ${data.message} (Trained on ${data.samples} tickets)`);
        } else {
            alert(`Error: ${data.detail || 'Could not train model'}`);
        }
    } catch (e) {
        alert('Network error: ' + e.message);
    } finally {
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}

// ---- Select and Delete ----
function toggleDeleteBtn() {
    const checked = document.querySelectorAll('.ticket-checkbox:checked').length;
    const btn = document.getElementById('deleteSelectedBtn');
    btn.disabled = checked === 0;
    btn.style.opacity = checked === 0 ? '0.5' : '1';
}

async function deleteSelected() {
    const checkedBoxes = Array.from(document.querySelectorAll('.ticket-checkbox:checked'));
    const ticketIds = checkedBoxes.map(cb => Number(cb.value));
    
    if (!ticketIds.length) return;
    if (!confirm(`Are you sure you want to delete ${ticketIds.length} selected ticket(s)?`)) return;
    
    try {
        const res = await fetch(`${API}/api/tickets/bulk`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticket_ids: ticketIds })
        });
        
        if (res.ok) {
            const data = await res.json();
            alert(data.message);
            loadTickets();
            loadStats();
        } else {
            alert('Error deleting tickets.');
        }
    } catch (e) {
        alert('Network error: ' + e.message);
    }
}

// ---- Keyboard shortcut ----
document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        submitComplaint();
    }
});

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    loadTickets();
    loadStats();
});
