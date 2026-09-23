/* 购物助手 Agent · 界面逻辑
   没有用前端框架。这个工程的定位是「解开就能读」，多一个构建步骤就多一道门槛。
   代码分四块：基础请求、对话流、记忆面板、设置弹窗。 */

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
  (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

const state = {
  userId: 'U1001',
  sessionId: newSid(),
  config: {},
  presets: {},
  users: [],
  streaming: false,
  memTab: 'profile',
  mem: null,
  trace: [],
};

function newSid() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, '0');
  return `S${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${Math.random().toString(36).slice(2, 7)}`;
}

/* ── 基础请求 ──────────────────────────────────────────── */
async function api(path, body) {
  const opt = body
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    : {};
  const res = await fetch(path, opt);
  return res.json();
}

/* SSE 解析。浏览器原生的 EventSource 只支持 GET，而对话要带一段 JSON 过去，
   所以用 fetch 读流自己拆帧。帧格式 = 若干 `字段: 值` 行 + 一个空行。

   两个字段都要看：服务端收尾会发一条 `event: end`，它不带业务内容。
   只看 data: 行的话会把这条收尾帧当成一个空事件，界面上就多出一条没有标题的步骤。 */
async function stream(path, body, onEvent) {
  let res;
  try {
    res = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } catch (e) {
    onEvent({ kind: 'error', message: '连不上服务，检查一下终端窗口还在不在' });
    return;
  }
  if (!res.ok) {
    onEvent({ kind: 'error', message: `服务返回 HTTP ${res.status}` });
    return;
  }
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  let ended = false;
  while (!ended) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, i);
      buf = buf.slice(i + 2);
      let name = '';
      let data = '';
      for (const line of frame.split('\n')) {
        if (line.startsWith(':')) continue;              // 心跳注释行
        if (line.startsWith('event: ')) name = line.slice(7).trim();
        else if (line.startsWith('data: ')) data = line.slice(6);
      }
      if (name === 'end') { ended = true; break; }       // 收尾帧，没有业务内容
      if (!data) continue;
      let ev;
      try { ev = JSON.parse(data); } catch (_) { continue; }
      if (!ev || typeof ev !== 'object' || !ev.kind) continue;
      if (ev.kind === '__end__') { ended = true; break; }
      onEvent(ev);
    }
  }
}

/* ── 启动 ──────────────────────────────────────────────── */
async function boot() {
  const d = await api('/api/bootstrap');
  state.config = d.config;
  state.presets = d.presets;
  state.users = d.users;
  state.stats = d.stats;

  $('#userSelect').innerHTML = d.users.map((u) =>
    `<option value="${u.user_id}">${esc(u.nickname)} · ${esc(u.user_id)} · ${esc(u.city)}</option>`
  ).join('');
  $('#userSelect').value = state.userId;
  paintMode();
  paintExamples();
  await loadMemory();
  await loadSession();
}

function paintMode() {
  const live = state.config.mode === 'live';
  const b = $('#modeBadge');
  b.textContent = live
    ? `真实模型 · ${state.config.model || '未指定'}`
    : '内置规则（还没配模型）';
  b.className = 'badge' + (live ? ' live' : '');
}

const EXAMPLES = [
  '想买个通勤坐地铁用的降噪耳机，预算一千以内',
  '聆风 A2 Pro 和木石 M9 哪个降噪更好',
  '我的订单到哪了',
  '退货运费谁承担',
  '我对夹头敏感，戴眼镜，帮我避开这种设计',
  '上次我提过什么要求？',
];

function paintExamples() {
  $('#examples').innerHTML = EXAMPLES.map((t) =>
    `<button data-q="${esc(t)}">${esc(t)}</button>`).join('');
  $$('#examples button').forEach((b) => {
    b.onclick = () => { $('#input').value = b.dataset.q; send(); };
  });
}

/* ── 对话 ──────────────────────────────────────────────── */
let curBubble = null;

function addUserMsg(text) {
  const el = document.createElement('div');
  el.className = 'msg user';
  el.innerHTML = `<div class="who">你</div><div class="bubble">${esc(text)}</div>`;
  $('#messages').appendChild(el);
  scrollDown();
}

function ensureBubble() {
  if (curBubble) return curBubble;
  const el = document.createElement('div');
  el.className = 'msg bot';
  el.innerHTML = `<div class="who"><b>助手</b></div><div class="bubble pending cursor"></div>`;
  $('#messages').appendChild(el);
  curBubble = $('.bubble', el);
  scrollDown();
  return curBubble;
}

function appendToken(text) {
  const b = ensureBubble();
  b.classList.remove('pending');
  b.textContent += text;
  scrollDown();
}

function appendStep(ev) {
  const kind = ev.kind || 'step';
  const el = document.createElement('div');
  el.className = 'step';
  el.dataset.kind = kind;
  if (ev.write) el.dataset.write = '1';

  const tags = { route: '路由', tool: '工具', memory: '记忆', archive: '归档',
                 govern: '上下文', guard: '护栏', halt: '熔断', chat: '兜底',
                 think: '思考' };
  const tag = tags[kind] || (ev.role || '');
  const ms = ev.ms ? `<span class="st-ms">${ev.ms} ms</span>` : '';

  let detail = ev.detail || '';
  if (ev.args && Object.keys(ev.args).length) {
    detail += (detail ? '\n' : '') + '参数：' + JSON.stringify(ev.args, null, 0);
  }

  el.innerHTML = `
    <div class="st-head">
      ${tag ? `<span class="st-tag">${esc(tag)}</span>` : ''}
      <span>${esc(ev.title || '')}</span>${ms}
    </div>
    ${detail ? `<details><summary>展开</summary><div class="st-detail">${esc(detail)}</div></details>` : ''}
    ${detail ? `<div class="st-detail" style="display:none"></div>` : ''}`;

  // 短内容直接显示，长的折起来
  const det = $('details', el);
  const inline = $('.st-detail', el);
  if (detail && detail.length <= 120) {
    inline.textContent = detail;
    inline.style.display = 'block';
    if (det) det.remove();
  }

  const anchor = curBubble ? curBubble.closest('.msg') : null;
  if (anchor) $('#messages').insertBefore(el, anchor);
  else $('#messages').appendChild(el);

  state.trace.push(ev);
  paintTrace();
  scrollDown();
}

function showConfirm(ev) {
  const p = ev.payload || {};
  const el = document.createElement('div');
  el.className = 'confirm';
  el.innerHTML = `
    <div class="cf-head"><span>${esc(p.title || '需要确认')}</span><span>金额确认</span></div>
    <div class="cf-body">
      <div>${esc(p.product || p.sku_id)} × ${p.qty}　（${esc(p.sku_id)}）</div>
      <table>
        <tr><td>单价</td><td>${money(p.unit_price)}</td></tr>
        <tr><td>小计</td><td>${money(p.subtotal)}</td></tr>
        <tr><td>优惠${p.applied && p.applied.length ? '（' + esc(p.applied.join('、')) + '）' : ''}</td><td>-${money(p.discount)}</td></tr>
        <tr class="total"><td>应付</td><td>${money(p.payable)}</td></tr>
      </table>
      <div style="color:#8a8f9a;font-size:12.5px">${esc(p.reason || '')}</div>
    </div>
    <div class="cf-actions">
      <button class="primary" data-ok="1">确认下单</button>
      <button class="ghost" data-ok="0">取消</button>
    </div>`;
  $('#messages').appendChild(el);
  $$('.cf-actions button', el).forEach((b) => {
    b.onclick = () => {
      el.classList.add('done');
      $$('.cf-actions button', el).forEach((x) => x.disabled = true);
      const ok = b.dataset.ok === '1';
      el.querySelector('.cf-head').lastElementChild.textContent = ok ? '已确认' : '已取消';
      stream('/api/confirm', { session_id: state.sessionId, user_id: state.userId, approved: ok },
        handleEvent);
      setBusy(true);
    };
  });
  scrollDown();
}

function money(v) { return '¥' + Number(v || 0).toFixed(2); }

function handleEvent(ev) {
  switch (ev.kind) {
    case 'token':    appendToken(ev.text); break;
    case 'final': {
      const shown = curBubble ? curBubble.textContent : '';
      const real = ev.reply || '';
      if (curBubble) {
        curBubble.classList.remove('pending', 'cursor');
        /* 逐字吐出去的内容，和最终答复可能不一样。
           护栏会把「没有凭据的话」换掉 —— 比如模型没调下单工具却说
           「已为您下单」。那些字已经显示在屏幕上了，不能就这么留着，
           否则护栏拦了等于没拦。以最终答复为准，覆盖掉，并标一下。 */
        if (real && real !== shown) {
          curBubble.textContent = real;
          curBubble.dataset.corrected = '1';
          appendStep({ kind: 'guard', title: '答复已更正',
                       detail: '流式显示的内容与最终答复不一致，'
                             + '已按经过校验的答复替换' });
        }
      }
      curBubble = null;
      setBusy(false);
      if (ev.usage) {
        $('#usageLine').textContent = ev.usage.calls
          ? `本轮模型调用 ${ev.usage.calls} 次 · 输入 ${ev.usage.prompt_tokens} · 输出 ${ev.usage.completion_tokens} token`
          : '本轮没有调用模型';
      }
      finishRound(ev);
      break;
    }
    case 'awaiting':
      setBusy(false);
      $('#statusLine').textContent = '等你在上面确认';
      break;
    case 'confirm':  showConfirm(ev); break;
    case 'error':
      curBubble = null;
      setBusy(false);
      appendStep({ kind: 'guard', title: '出错了', detail: ev.message });
      break;
    default:
      appendStep(ev);
  }
}

async function finishRound() {
  await loadMemory();
  await loadSession(true);
}

function setBusy(on) {
  state.streaming = on;
  $('#btnSend').disabled = on;
  $('#statusLine').textContent = on ? '正在处理…' : '就绪';
}

async function send() {
  const text = $('#input').value.trim();
  if (!text || state.streaming) return;
  $('#input').value = '';
  $('#input').style.height = 'auto';
  addUserMsg(text);
  curBubble = null;
  state.trace = [];
  paintTrace();
  setBusy(true);
  $('#usageLine').textContent = '';
  await stream('/api/chat',
    { session_id: state.sessionId, user_id: state.userId, text }, handleEvent);
  setBusy(false);
}

/* ── 记忆面板 ──────────────────────────────────────────── */
async function loadMemory() {
  state.mem = await api(`/api/memory?user_id=${encodeURIComponent(state.userId)}`);
  const s = state.mem.stats;
  $('#memStats').textContent =
    `画像 ${s.profile_active}${s.profile_muted ? '(+' + s.profile_muted + '已停用)' : ''} · 事件 ${s.episodes}`;
  paintMemory();
}

function paintMemory() {
  const box = $('#memBody');
  const m = state.mem;
  if (!m) { box.innerHTML = '<p class="empty">载入中…</p>'; return; }

  if (state.memTab === 'profile') {
    if (!m.profile.length) {
      box.innerHTML = `<p class="empty">还没有画像记忆。<br><br>
        试着说一句「我对夹头敏感，戴眼镜」，或者「我不喜欢云雀这个牌子」。</p>`;
      return;
    }
    box.innerHTML = m.profile.map((p) => `
      <div class="mem-item ${p.active ? '' : 'muted'}">
        <button class="mi-del" data-key="${esc(p.key)}" title="忘掉这条">×</button>
        <div class="mi-top">
          <span class="mi-kind ${esc(p.polarity)}">${esc(p.kind)}</span>
          <span class="mi-val">${esc(p.value)}</span>
        </div>
        <div class="mi-meta">
          置信度 ${(p.confidence * 100).toFixed(0)}%　证据 ${p.evidence_count} 次　
          ${p.active ? `命中 ${p.hits} 次` : '已停用'}<br>
          ${p.last_hit || p.updated_at}　${esc((p.quote || '').slice(0, 40))}
        </div>
        <div class="bar"><i style="width:${Math.round(p.confidence * 100)}%"></i></div>
      </div>`).join('');
    $$('.mi-del', box).forEach((b) => {
      b.onclick = async () => {
        await api('/api/memory/forget', { user_id: state.userId, key: b.dataset.key });
        await loadMemory();
      };
    });

  } else if (state.memTab === 'episodes') {
    if (!m.episodes.length) {
      box.innerHTML = '<p class="empty">还没有事件记忆。聊过订单、物流、退换货之后这里会记下来。</p>';
      return;
    }
    box.innerHTML = m.episodes.map((e) => `
      <div class="mem-item">
        <button class="mi-del" data-ep="${esc(e.ep_id)}" title="删掉这条">×</button>
        <div class="mi-top"><span class="mi-kind">${esc(e.kind)}</span></div>
        <div>${esc(e.summary)}</div>
        <div class="mi-meta">
          ${esc(e.created_at)}　${e.age_days} 天前　权重 ${e.decay}　命中 ${e.hits} 次
          ${e.order_id ? '　订单 ' + esc(e.order_id) : ''}
        </div>
      </div>`).join('');
    $$('.mi-del', box).forEach((b) => {
      b.onclick = async () => {
        await api('/api/memory/drop-episode', { ep_id: b.dataset.ep });
        await loadMemory();
      };
    });

  } else {
    if (!m.sessions.length) {
      box.innerHTML = '<p class="empty">这个身份下还没有会话记录。</p>';
      return;
    }
    box.innerHTML = m.sessions.map((s) => `
      <div class="mem-item">
        <div class="mi-top">
          <span class="mi-kind">${s.turns} 轮</span>
          <span class="mi-val">${esc(s.title || '(无标题)')}</span>
        </div>
        <div class="mi-meta">${esc(s.updated_at)}</div>
        ${s.summary ? `<div class="mi-meta">摘要：${esc(s.summary)}</div>` : ''}
      </div>`).join('');
  }
}

/* ── 会话历史 ──────────────────────────────────────────── */
async function loadSession() {
  const d = await api(`/api/session?session_id=${encodeURIComponent(state.sessionId)}&user_id=${encodeURIComponent(state.userId)}`);
  if (d.history && d.history.length) paintHistory(d.history);
}

function paintHistory(history) {
  $('#messages').innerHTML = '';
  for (const m of history) {
    const el = document.createElement('div');
    el.className = 'msg ' + (m.role === 'user' ? 'user' : 'bot');
    el.innerHTML = `<div class="who">${m.role === 'user' ? '你' : '<b>助手</b>'}</div>
      <div class="bubble">${esc(m.content)}</div>`;
    $('#messages').appendChild(el);
  }
  scrollDown();
}

/* ── 轨迹面板 ──────────────────────────────────────────── */
function paintTrace() {
  const box = $('#traceBody');
  if (!state.trace.length) {
    box.innerHTML = '<p class="empty">还没有对话。</p>';
    return;
  }
  box.innerHTML = '<div class="tl">' + state.trace.map((e) => `
    <div class="tl-item" data-kind="${esc(e.kind || '')}">
      <div class="tl-t">${esc(e.title || '')}<span>${e.ms ? e.ms + ' ms' : ''}</span></div>
      <div class="tl-d">${esc((e.detail || '').slice(0, 160))}</div>
    </div>`).join('') + '</div>';
}

function scrollDown() {
  const m = $('#messages');
  m.scrollTop = m.scrollHeight;
}

/* ── 设置弹窗 ──────────────────────────────────────────── */
function openSettings() {
  const c = state.config;
  $('#cfgBaseUrl').value = c.base_url || '';
  $('#cfgApiKey').value = c.api_key || '';
  $('#cfgModel').value = c.model || '';
  $('#cfgTemp').value = c.temperature;
  $('#cfgMaxTokens').value = c.max_tokens;
  $('#cfgTimeout').value = c.timeout;
  $('#cfgMaxRounds').value = c.max_rounds;
  $('#cfgMaxAmount').value = c.max_amount;
  $('#cfgHistory').value = c.history_limit;
  $('#cfgMemory').value = c.memory_limit;
  $('#testResult').innerHTML = '';
  paintPresets();
  $('#settingsModal').hidden = false;
}

function paintPresets() {
  const cur = state.config.provider;
  $('#presetRow').innerHTML = Object.entries(state.presets).map(([k, v]) =>
    `<button data-p="${k}" class="${k === cur ? 'active' : ''}"
       title="${esc(v.hint)}">${esc(v.label)}</button>`).join('');
  $$('#presetRow button').forEach((b) => {
    b.onclick = () => {
      const p = state.presets[b.dataset.p];
      $('#cfgBaseUrl').value = p.base_url;
      $('#cfgModel').value = p.model || '';
      if (p.api_key) $('#cfgApiKey').value = p.api_key;
      state.config.provider = b.dataset.p;
      paintPresets();
      $('#testResult').innerHTML = `<div class="models">${esc(p.hint)}</div>`;
    };
  });
}

function formValues() {
  return {
    provider: state.config.provider || 'lmstudio',
    base_url: $('#cfgBaseUrl').value.trim(),
    api_key: $('#cfgApiKey').value,
    model: $('#cfgModel').value.trim(),
    temperature: parseFloat($('#cfgTemp').value) || 0,
    max_tokens: parseInt($('#cfgMaxTokens').value, 10) || 900,
    timeout: parseInt($('#cfgTimeout').value, 10) || 120,
    max_rounds: parseInt($('#cfgMaxRounds').value, 10) || 6,
    max_amount: parseFloat($('#cfgMaxAmount').value) || 2000,
    history_limit: parseInt($('#cfgHistory').value, 10) || 12,
    memory_limit: parseInt($('#cfgMemory').value, 10) || 6,
  };
}

async function loadModels() {
  const box = $('#testResult');
  box.innerHTML = '<div class="models">正在连服务…</div>';
  const r = await api('/api/models?' + new URLSearchParams({
    base_url: $('#cfgBaseUrl').value.trim(),
    api_key: $('#cfgApiKey').value,
    timeout: $('#cfgTimeout').value,
  }));
  if (!r.models || !r.models.length) {
    box.innerHTML = `<div class="bad">${esc(r.message || '没拿到模型列表')}</div>`;
    return;
  }
  $('#modelOptions').innerHTML = r.models.map((m) => `<option value="${esc(m)}">`).join('');
  if (!$('#cfgModel').value) $('#cfgModel').value = r.models[0];
  box.innerHTML = `<div class="ok">拿到 ${r.models.length} 个模型</div>
    <div class="models">${esc(r.models.slice(0, 8).join('、'))}${r.models.length > 8 ? ' …' : ''}</div>`;
}

async function testConn() {
  const box = $('#testResult');
  box.innerHTML = '<div class="models">正在测试。本地模型首次加载可能要十几秒…</div>';
  const r = await api('/api/config/test', formValues());
  if (r.models && r.models.length) {
    $('#modelOptions').innerHTML = r.models.map((m) => `<option value="${esc(m)}">`).join('');
  }
  box.innerHTML = `<div class="${r.ok ? 'ok' : 'bad'}">${esc(r.message)}</div>
    ${r.sample ? `<div class="models">模型回了一句：${esc(r.sample)}</div>` : ''}
    ${r.models && r.models.length ? `<div class="models">可选模型 ${r.models.length} 个</div>` : ''}`;
}

async function saveConfig() {
  const r = await api('/api/config', formValues());
  state.config = r.config;
  paintMode();
  $('#settingsModal').hidden = true;
  appendStep({ kind: 'govern', title: '设置已保存',
    detail: `当前模式：${r.config.mode === 'live' ? '真实模型 ' + r.config.model : '内置规则'}。已建的会话会用新配置重建。` });
}

/* ── 绑定 ──────────────────────────────────────────────── */
function bind() {
  $('#btnSend').onclick = send;
  $('#input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
  $('#input').addEventListener('input', (e) => {
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 160) + 'px';
  });

  $('#userSelect').onchange = async (e) => {
    state.userId = e.target.value;
    state.sessionId = newSid();
    $('#messages').innerHTML = '';
    state.trace = [];
    paintTrace();
    await loadMemory();
  };

  $('#btnNew').onclick = () => {
    state.sessionId = newSid();
    $('#messages').innerHTML = '';
    state.trace = [];
    paintTrace();
    $('#usageLine').textContent = '';
    appendStep({ kind: 'govern', title: '开了个新会话',
      detail: '短期历史清空了。长期记忆还在，换个会话它照样记得你。' });
  };

  $('#memTabs').onclick = (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    state.memTab = b.dataset.tab;
    $$('#memTabs button').forEach((x) => x.classList.toggle('active', x === b));
    paintMemory();
  };

  $('#btnDecay').onclick = async () => {
    const r = await api('/api/memory/decay', { user_id: state.userId });
    await loadMemory();
    appendStep({ kind: 'archive', title: '遗忘扫描完成',
      detail: r.muted ? `停用了 ${r.muted} 条长期没用上的画像（数据没删，数据还在）`
                      : '没有需要停用的条目。置信度低且长期没被用上的画像才会被停用。' });
  };

  $('#btnClearMem').onclick = async () => {
    if (!confirm('清空这个身份的全部记忆？画像、事件、会话摘要都会删掉。')) return;
    await api('/api/memory/clear', { user_id: state.userId, layer: 'all' });
    await loadMemory();
    appendStep({ kind: 'archive', title: '记忆已清空', detail: `已清空 ${state.userId} 的三层记忆。` });
  };

  $('#btnToggleTrace').onclick = (e) => {
    const p = $('#rightPanel');
    const hidden = p.style.display === 'none';
    p.style.display = hidden ? '' : 'none';
    e.target.textContent = hidden ? '收起' : '展开';
    document.querySelector('main').style.gridTemplateColumns =
      hidden ? '' : '296px minmax(0, 1fr)';
  };

  $('#btnSettings').onclick = openSettings;
  $('#btnCloseSettings').onclick = () => { $('#settingsModal').hidden = true; };
  $('#settingsModal').onclick = (e) => {
    if (e.target.id === 'settingsModal') $('#settingsModal').hidden = true;
  };
  $('#btnLoadModels').onclick = loadModels;
  $('#btnTest').onclick = testConn;
  $('#btnSave').onclick = saveConfig;

  $('#btnLoadModels').click = loadModels;
  window.addEventListener('resize', () => {
    const list = $('#modelOptions');
    if (list && !list.children.length && state.config.mode === 'live') loadModels();
  });
}

bind();
boot();
