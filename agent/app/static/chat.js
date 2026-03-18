// 商品选购助手 - 对话与历史管理
const API_BASE = '';

let currentSessionId = null;
let sessions = [];

const sidebar = document.getElementById('sidebar-body');
const messagesContainer = document.getElementById('messages-container');
const emptyState = document.getElementById('empty-state');
const queryForm = document.getElementById('query-form');
const keywordInput = document.getElementById('keyword-input');
const budgetMaxInput = document.getElementById('budget-max-input');
const usageInput = document.getElementById('usage-input');
const budgetMinInput = document.getElementById('budget-min-input');
const brandPrefInput = document.getElementById('brand-pref-input');
const brandAvoidInput = document.getElementById('brand-avoid-input');
const portabilityInput = document.getElementById('portability-input');
const gpuInput = document.getElementById('gpu-input');
const extraInput = document.getElementById('extra-input');
const sendBtn = document.getElementById('send-btn');
const newChatBtn = document.getElementById('new-chat-btn');

// 初始化
async function init() {
  await loadSessions();
  newChatBtn.addEventListener('click', createNewChat);
  queryForm.addEventListener('submit', (e) => {
    e.preventDefault();
    sendMessage();
  });
}

// 加载会话列表
async function loadSessions() {
  try {
    const res = await fetch(`${API_BASE}/api/sessions`);
    const data = await res.json();
    sessions = data.sessions || [];
    renderSessions();
  } catch (err) {
    console.error('加载会话失败', err);
    sessions = [];
  }
}

// 渲染会话列表
function renderSessions() {
  sidebar.innerHTML = sessions
    .map((s) => `
      <div class="session-item ${s.id === currentSessionId ? 'active' : ''}" data-id="${s.id}">
        <span class="title">${escapeHtml(s.title || '新对话')}</span>
        <button class="delete-btn" data-id="${s.id}" title="删除">×</button>
      </div>
    `)
    .join('');
  sidebar.querySelectorAll('.session-item').forEach((el) => {
    el.addEventListener('click', (e) => {
      if (e.target.classList.contains('delete-btn')) return;
      selectSession(el.dataset.id);
    });
    const del = el.querySelector('.delete-btn');
    if (del) del.addEventListener('click', (e) => {
      e.stopPropagation();
      deleteSession(del.dataset.id);
    });
  });
}

// 创建新对话
async function createNewChat() {
  try {
    const res = await fetch(`${API_BASE}/api/sessions`, { method: 'POST' });
    const data = await res.json();
    const session = data.session;
    sessions.unshift(session);
    renderSessions();
    selectSession(session.id);
  } catch (err) {
    console.error('创建会话失败', err);
  }
}

// 选择会话
async function selectSession(id) {
  currentSessionId = id;
  renderSessions();
  try {
    const res = await fetch(`${API_BASE}/api/sessions/${id}`);
    const data = await res.json();
    const messages = data.messages || [];
    renderMessages(messages);
  } catch (err) {
    console.error('加载会话失败', err);
  }
}

// 删除会话
async function deleteSession(id) {
  try {
    await fetch(`${API_BASE}/api/sessions/${id}`, { method: 'DELETE' });
    sessions = sessions.filter((s) => s.id !== id);
    if (currentSessionId === id) {
      currentSessionId = null;
      renderMessages([]);
    }
    renderSessions();
  } catch (err) {
    console.error('删除会话失败', err);
  }
}

// 发送消息
async function sendMessage() {
  const text = buildQueryTextFromForm();
  if (!text) return;

  if (!currentSessionId) {
    await createNewChat();
    if (!currentSessionId) return;
  }

  const userMsg = { role: 'user', content: text };
  appendMessage(userMsg);
  setInputState(true);

  appendMessage({ role: 'assistant', content: '', loading: true });

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: currentSessionId,
        message: text,
      }),
    });
    const data = await res.json();
    if (data.session_id) currentSessionId = data.session_id;

    removeLoadingMessage();
    const formatted = data.formatted_response || formatResponse(data.response);
    appendMessage({ role: 'assistant', content: formatted, raw: data.response });
  } catch (err) {
    removeLoadingMessage();
    appendMessage({
      role: 'assistant',
      content: `请求失败：${err.message}`,
    });
  } finally {
    setInputState(false);
  }
}

function buildQueryTextFromForm() {
  const keyword = keywordInput.value.trim();
  const budgetMax = budgetMaxInput.value.trim();
  const usage = usageInput.value.trim();
  const budgetMin = budgetMinInput.value.trim();
  const brandPref = brandPrefInput.value.trim();
  const brandAvoid = brandAvoidInput.value.trim();
  const portability = portabilityInput.value;
  const gpu = gpuInput.value;
  const extra = extraInput.value.trim();

  if (!keyword || !budgetMax || !usage) {
    alert('请先填写必填项：品类、预算上限、使用场景。');
    return '';
  }

  const parts = [
    `我想买${keyword}`,
    budgetMin ? `预算在${budgetMin}-${budgetMax}元` : `预算上限${budgetMax}元`,
    `主要用于${usage}`,
  ];
  if (brandPref) parts.push(`品牌偏好：${brandPref}`);
  if (brandAvoid) parts.push(`不考虑品牌：${brandAvoid}`);
  if (portability === 'true') parts.push('需要便携轻薄');
  if (portability === 'false') parts.push('不强调便携');
  if (gpu === 'true') parts.push('需要独立显卡');
  if (gpu === 'false') parts.push('不需要独立显卡');
  if (extra) parts.push(`其他要求：${extra}`);
  return `${parts.join('，')}。`;
}

// 格式化 AgentResponse 为可读内容
function formatResponse(res) {
  if (!res) return '暂无推荐结果';
  const parts = [];

  if (res.clarifying_questions && res.clarifying_questions.length > 0) {
    parts.push('**需要补充的信息：**\n');
    res.clarifying_questions.forEach((q) => parts.push(`- ${q}`));
    parts.push('');
  }

  if (res.final_recommendation) {
    const p = res.final_recommendation;
    parts.push(`## 推荐商品：${p.name}\n`);
    parts.push(`**价格：** ¥${p.price}\n`);
    if (p.brand) parts.push(`**品牌：** ${p.brand}\n`);
    if (p.cpu_model || p.gpu_model) {
      const specs = [p.cpu_model, p.gpu_model].filter(Boolean).join(' / ');
      parts.push(`**配置：** ${specs}\n`);
    }
    if (p.memory_gb || p.storage_gb) {
      parts.push(`**内存/存储：** ${p.memory_gb}GB / ${p.storage_gb}GB\n`);
    }
    if (p.purchase_url) {
      parts.push(`\n[点击购买](${p.purchase_url})\n`);
    }
    if (res.recommendation_reason && res.recommendation_reason.length > 0) {
      parts.push('\n**推荐理由：**\n');
      res.recommendation_reason.forEach((r) => parts.push(`- ${r}`));
    }
  } else if (res.candidates && res.candidates.length > 0) {
    parts.push('**候选商品：**\n');
    res.candidates.slice(0, 5).forEach((p, i) => {
      parts.push(`${i + 1}. ${p.name} - ¥${p.price}`);
    });
  } else {
    parts.push('暂未找到符合条件的商品，请补充更多需求后重试。');
  }

  return parts.join('\n');
}

// 简单 Markdown 转 HTML
function mdToHtml(text) {
  if (!text) return '';
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/^## (.+)$/gm, '<h4>$1</h4>')
    .replace(/^- (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
    .replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank">$1</a>')
    .replace(/\n/g, '<br>');
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// 渲染消息列表
function renderMessages(messages) {
  messagesContainer.innerHTML = '';
  emptyState.style.display = messages.length ? 'none' : 'flex';
  messages.forEach((m) => appendMessage(m, false));
}

// 追加一条消息
function appendMessage(msg, scroll = true) {
  emptyState.style.display = 'none';
  const div = document.createElement('div');
  div.className = `message ${msg.role}`;
  const label = msg.role === 'user' ? '你' : '助手';
  if (msg.loading) {
    div.innerHTML = `<span class="role-label">${label}</span><div class="typing-indicator"><span></span><span></span><span></span></div>`;
  } else {
    const content = msg.content ? mdToHtml(msg.content) : '';
    div.innerHTML = `<span class="role-label">${label}</span><div class="content">${content}</div>`;
  }
  messagesContainer.appendChild(div);
  if (scroll) messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function removeLoadingMessage() {
  const last = messagesContainer.querySelector('.message.assistant:last-child');
  if (last && last.querySelector('.typing-indicator')) last.remove();
}

function setInputState(disabled) {
  queryForm.querySelectorAll('input, select, button').forEach((el) => {
    el.disabled = disabled;
  });
  sendBtn.disabled = disabled;
}

// 若没有会话则创建
init().then(() => {
  if (sessions.length === 0) createNewChat();
});
