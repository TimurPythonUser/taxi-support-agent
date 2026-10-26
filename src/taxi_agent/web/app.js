"use strict";

const $ = (id) => document.getElementById(id);
const els = {
  messages: $("messages"), empty: $("empty"), input: $("input"), send: $("send"), composer: $("composer"),
  receipt: $("receipt"), caseNote: $("case-note"), handoff: $("handoff"), trace: $("trace"),
  tickets: $("tickets"), ticketsCount: $("tickets-count"), status: $("status-pill"), meta: $("chat-meta"),
  mIn: $("m-in"), mCache: $("m-cache"), mOut: $("m-out"), mUsd: $("m-usd"),
};
const STORAGE_KEY = "taxi-agent-session";

const SCENARIOS = {
  lost_item: {
    title: "Забытая вещь",
    fields: [
      ["order_id", "Номер заказа"], ["trip_date", "Дата поездки"], ["approx_time", "Время"],
      ["route", "Маршрут"], ["item_description", "Вещь"], ["contact_phone", "Телефон"],
    ],
  },
  trip_price: {
    title: "Стоимость поездки",
    fields: [["origin", "Откуда"], ["destination", "Куда"], ["tariff", "Тариф"]],
  },
  driver_complaint: {
    title: "Жалоба на водителя",
    fields: [
      ["order_id", "Номер заказа"], ["trip_date", "Дата поездки"], ["approx_time", "Время"],
      ["route", "Маршрут"], ["reason_category", "Причина"], ["description", "Что случилось"],
      ["safety_incident", "Угроза безопасности"],
    ],
  },
};
const MISSING_TO_FIELDS = {
  order_id_or_trip_date: ["order_id", "trip_date"],
  order_id_or_approx_time: ["order_id", "approx_time"],
};
const TARIFFS = { economy: "Эконом", comfort: "Комфорт", business: "Бизнес" };
const REASONS = {
  rude_behavior: "грубость", dangerous_driving: "опасное вождение", route_or_fare: "маршрут / цена",
  car_condition: "состояние авто", late_or_no_show: "опоздание", safety_threat: "угроза безопасности", other: "другое",
};
const KINDS = { lost_item: "забытая вещь", driver_complaint: "жалоба", handoff: "передача оператору" };
const PRIORITIES = { normal: "обычный", high: "повышенный", urgent: "срочно" };
const GUARDS = {
  human_requested: "клиент попросил оператора — передано без модели",
  safety_keywords: "слова об угрозе безопасности — включён режим тревоги",
  unverified_price: "в ответе была цена не из сервиса — ответ переписан",
  safety_forced_handoff: "срочная передача оператору (безопасность)",
  stalled_handoff: "нет прогресса несколько реплик — передано оператору",
  technical_handoff: "повторный сбой — передано оператору",
  already_handed_off: "диалог уже у оператора — фиксированный ответ",
  session_limit: "лимит сообщений демо-диалога",
  daily_limit: "дневной лимит демо-стенда",
  llm: "ошибка модели",
  tool_loop_limit: "слишком много вызовов подряд — ответ без инструментов",
};

let sessionId = null;
let turnNo = 0;
let busy = false;

const storage = {
  get() { try { return localStorage.getItem(STORAGE_KEY); } catch { return null; } },
  set(v) { try { localStorage.setItem(STORAGE_KEY, v); } catch { /* private mode */ } },
  clear() { try { localStorage.removeItem(STORAGE_KEY); } catch { /* private mode */ } },
};

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
function richText(s) {
  return esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}
function fmtDate(iso) {
  if (!iso) return "";
  const [y, m, d] = iso.split("-");
  return `${d}.${m}.${y}`;
}
function fmtValue(key, value) {
  if (value === null || value === undefined || value === "") return null;
  if (key === "tariff") return TARIFFS[value] || value;
  if (key === "reason_category") return REASONS[value] || value;
  if (key === "trip_date") return fmtDate(value);
  if (typeof value === "boolean") return value ? "да" : "нет";
  return String(value);
}

async function api(path, options = {}) {
  const resp = await fetch(path, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  if (!resp.ok) {
    const err = new Error(`HTTP ${resp.status}`);
    err.status = resp.status;
    throw err;
  }
  return resp.json();
}

// ---------- chat ----------
function addMessage(role, text) {
  els.empty.hidden = true;
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.innerHTML = role === "agent" ? `<span class="who">агент</span>${richText(text)}` : esc(text);
  els.messages.appendChild(div);
  els.messages.scrollTop = els.messages.scrollHeight;
  return div;
}

function showTyping() {
  const div = addMessage("agent typing", "");
  div.innerHTML = '<span class="who">агент</span><span class="dots"><span></span><span></span><span></span></span>';
  return div;
}

function setBusy(value) {
  busy = value;
  els.send.disabled = value;
  els.input.disabled = value;
  if (value) {
    els.status.dataset.state = "busy";
    els.status.textContent = "печатает…";
  }
}

async function send(text) {
  text = text.trim();
  if (!text || busy) return;
  addMessage("user", text);
  els.input.value = "";
  autosize();
  setBusy(true);
  const typing = showTyping();
  try {
    if (!sessionId) await newSession();
    const data = await api(`/api/sessions/${sessionId}/messages`, { method: "POST", body: JSON.stringify({ text }) });
    typing.remove();
    addMessage("agent", data.reply);
    turnNo += 1;
    renderTrace(data.events, turnNo);
    renderSession(data);
  } catch (err) {
    typing.remove();
    if (err.status === 404) {
      // the session is gone (e.g. the server data was reset): start over and keep what the customer typed
      storage.clear();
      sessionId = null;
      resetView();
      addMessage("error", "Этот диалог больше не найден на сервере — начат новый. Отправьте сообщение ещё раз.");
      els.input.value = text;
      autosize();
    } else {
      const msg = err.status === 401 ? "Нужна авторизация — обновите страницу."
        : err.status === 422 ? "Сообщение пустое или слишком длинное."
        : "Не удалось связаться с сервером. Попробуйте ещё раз.";
      addMessage("error", msg);
    }
    els.status.dataset.state = "active";
    els.status.textContent = "агент на связи";
  } finally {
    setBusy(false);
    els.input.focus();
  }
}

// ---------- console ----------
function renderSession(data) {
  const { state } = data;
  els.meta.textContent = `сессия ${state.session_id.slice(0, 8)} · язык ${state.language} · реплик ${state.user_turns}`;
  const handedOff = state.status === "handed_off";
  els.status.dataset.state = handedOff ? "handed_off" : "active";
  els.status.textContent = handedOff ? "у оператора" : "агент на связи";
  renderReceipt(state, data.missing_fields);
  renderHandoff(state);
  renderTickets(data.tickets);
  renderMeter(data.cost);
}

function renderReceipt(state, missingNames) {
  const active = state.cases.find((c) => c.id === state.active_case_id);
  if (!active) {
    els.caseNote.textContent = "сценарий ещё не определён";
    return;
  }
  const spec = SCENARIOS[active.scenario];
  const handedOff = state.status === "handed_off";
  const missing = new Set(handedOff ? [] : missingNames.flatMap((m) => MISSING_TO_FIELDS[m] || [m]));
  const hasOrder = Boolean(active.order_id);
  els.caseNote.textContent = handedOff ? "дальше ведёт оператор"
    : active.status === "done" ? "дело закрыто" : `не хватает: ${missing.size}`;

  const status = active.ticket_id
    ? `<span class="status done">${esc(active.ticket_id)}</span>`
    : active.status === "done" ? '<span class="status done">рассчитано</span>' : '<span class="status">собираем</span>';
  let html = `<div class="receipt-title"><span>${spec.title}</span>${status}</div>`;

  for (const [key, label] of spec.fields) {
    const shown = fmtValue(key, active[key]);
    const notNeeded = hasOrder && (key === "trip_date" || key === "approx_time") && !shown;
    let cls = "row";
    let value = shown;
    if (!shown && missing.has(key) && !notNeeded) { cls += " missing"; value = "уточнить"; }
    else if (!shown) { cls += " empty-opt"; value = notNeeded ? "не нужно" : "—"; }
    html += `<div class="${cls}"><span class="k">${label}</span><span class="lead"></span><span class="v">${esc(value)}</span></div>`;
  }

  if (active.scenario === "trip_price" && active.quote) {
    const q = active.quote;
    html += `<div class="receipt-total"><div><div class="sub">предварительно · ${esc(TARIFFS[q.tariff] || q.tariff)}</div>
      <div class="sub">~${q.duration_min} мин · ${q.distance_km} км · действует 10 мин</div></div>
      <span class="price">${q.price} ${esc(q.currency)}</span></div>`;
  }
  if (active.found_trip) {
    const t = active.found_trip;
    const tag = active.trip_confirmed ? "подтверждена" : "ждёт подтверждения клиента";
    html += `<div class="trip-card"><b>Поездка #${esc(t.order_id)}</b> · ${fmtDate(t.date)} ${esc(t.time)} · ${tag}<br>
      ${esc(t.origin)} → ${esc(t.destination)}<br>${esc(t.driver_name)}, ${esc(t.car)} · ${t.price} ${esc(t.currency)}</div>`;
  } else if (active.trip_lookup_done && active.candidate_order_ids?.length > 1) {
    html += `<div class="trip-card">Подходят несколько поездок: ${active.candidate_order_ids.map((i) => "#" + esc(i)).join(", ")} — агент уточняет у клиента</div>`;
  } else if (active.trip_lookup_done) {
    html += '<div class="trip-card">В истории заказов поездка не найдена — оператор поищет вручную</div>';
  }

  const earlier = state.cases.filter((c) => c.id !== active.id);
  if (earlier.length) {
    html += `<div class="earlier">Ранее в диалоге: ${earlier.map((c) => `${SCENARIOS[c.scenario].title.toLowerCase()}${c.ticket_id ? " (" + esc(c.ticket_id) + ")" : ""}`).join(", ")}</div>`;
  }
  els.receipt.innerHTML = html;
}

function renderHandoff(state) {
  const h = state.handoff;
  if (!h) { els.handoff.hidden = true; return; }
  els.handoff.hidden = false;
  els.handoff.className = `alert ${h.priority === "urgent" ? "urgent" : ""}`;
  els.handoff.innerHTML = `<strong>Передано оператору · ${esc(h.ticket_id)}</strong> · приоритет: ${PRIORITIES[h.priority] || h.priority}<br>${esc(h.reason)}`;
}

function eventBrief(e) {
  const r = e.result || {};
  if (e.kind !== "tool") return GUARDS[e.name] || e.name;
  switch (e.name) {
    case "update_case": {
      const parts = Object.keys(r.applied || {}).map((k) => `+${k}`);
      for (const k of Object.keys(r.rejected || {})) parts.push(`✕ ${k}`);
      return parts.join(" ") || (r.ok === false ? r.error : "без изменений");
    }
    case "find_trip": return r.ok ? `найдено: ${(r.matches || []).length}` : r.error;
    case "get_trip_price": return r.ok ? `${r.quote.price} ${r.quote.currency}` : r.error;
    case "search_knowledge_base": return `«${e.arguments.query || ""}» → ${(r.results || []).length}`;
    case "create_operator_ticket": return r.ok ? r.ticket_id : r.error;
    case "handoff_to_operator": return `${r.ticket_id || ""} · ${e.arguments.priority || ""}`;
    default: return r.error || "";
  }
}

function renderTrace(events, turn) {
  const empty = els.trace.querySelector(".trace-empty");
  if (empty) empty.remove();
  const sep = document.createElement("li");
  sep.className = "turn-sep";
  sep.textContent = events.length ? `реплика ${turn}` : `реплика ${turn} · ответ без инструментов`;
  els.trace.appendChild(sep);
  for (const e of events) {
    const li = document.createElement("li");
    const failed = e.result && (e.result.ok === false || (e.result.rejected && Object.keys(e.result.rejected).length));
    li.className = `ev ${e.kind === "tool" ? (failed ? "fail" : "") : "guard"}`;
    const detail = JSON.stringify({ arguments: e.arguments, result: e.result }, null, 2);
    li.innerHTML = `<details><summary><span class="dot"></span><span class="name">${esc(e.kind === "tool" ? e.name : "guardrail")}</span>
      <span class="brief">${esc(eventBrief(e))}</span><span class="ms">${e.kind === "tool" ? e.ms + " мс" : ""}</span></summary>
      <pre>${esc(detail)}</pre></details>`;
    els.trace.appendChild(li);
  }
  els.trace.lastElementChild?.scrollIntoView({ block: "nearest" });
}

function renderTickets(tickets) {
  els.ticketsCount.textContent = String(tickets.length);
  if (!tickets.length) return;
  els.tickets.innerHTML = tickets.map((t) => `
    <article class="ticket">
      <div class="ticket-top"><span class="ticket-id">${esc(t.id)}</span><span class="ticket-kind">${esc(KINDS[t.kind] || t.kind)}</span>
        <span class="badge ${esc(t.priority)}">${esc(PRIORITIES[t.priority] || t.priority)}</span></div>
      <p>${esc(t.summary)}</p>
      <details><summary>данные для оператора</summary><pre>${esc(JSON.stringify(t.fields, null, 2))}</pre></details>
    </article>`).join("");
}

function renderMeter(cost) {
  const n = (x) => x.toLocaleString("ru-RU");
  els.mIn.textContent = n(cost.prompt_tokens);
  els.mCache.textContent = n(cost.cached_prompt_tokens);
  els.mOut.textContent = n(cost.completion_tokens);
  els.mUsd.textContent = `$${cost.usd.toFixed(4)}`;
}

// ---------- session lifecycle ----------
function resetView() {
  els.messages.querySelectorAll(".msg").forEach((m) => m.remove());
  els.empty.hidden = false;
  els.trace.innerHTML = '<li class="trace-empty">Пока пусто.</li>';
  els.receipt.innerHTML = '<div class="receipt-empty">Агент заполнит карточку, как только поймёт, с чем вы пришли. Всё, что здесь есть, — данные, которые клиент реально сообщил или вернули сервисы.</div>';
  els.caseNote.textContent = "сценарий ещё не определён";
  els.handoff.hidden = true;
  els.tickets.innerHTML = '<div class="tickets-empty">Агент создаёт обращение, только когда собраны все обязательные данные.</div>';
  els.ticketsCount.textContent = "0";
  els.meta.textContent = "новый диалог";
  els.status.dataset.state = "active";
  els.status.textContent = "агент на связи";
  renderMeter({ prompt_tokens: 0, cached_prompt_tokens: 0, completion_tokens: 0, usd: 0 });
  turnNo = 0;
}

async function newSession() {
  const data = await api("/api/sessions", { method: "POST" });
  sessionId = data.session_id;
  storage.set(sessionId);
  return data;
}

async function restore() {
  const saved = storage.get();
  if (!saved) return;
  try {
    const data = await api(`/api/sessions/${saved}`);
    sessionId = saved;
    for (const m of data.messages) addMessage(m.role === "user" ? "user" : "agent", m.content);
    turnNo = data.state.user_turns;
    if (turnNo) {
      els.trace.innerHTML = '<li class="trace-empty">Диалог восстановлен. Вызовы инструментов будут видны для новых реплик.</li>';
    }
    renderSession(data);
  } catch {
    storage.clear();
  }
}

async function loadDemoTrips() {
  const table = $("demo-trips");
  try {
    const trips = await api("/api/demo/trips");
    table.innerHTML = "<tbody>" + trips.map((t) => `<tr><td>${esc(t.order_id)}</td>
      <td class="when">${fmtDate(t.date).slice(0, 5)} ${esc(t.time)}</td>
      <td>${esc(t.origin)} → ${esc(t.destination)}<br><span style="color:var(--ink-3)">${esc(t.driver_name)}, ${esc(t.car)}</span></td></tr>`).join("") + "</tbody>";
  } catch {
    table.innerHTML = "<tbody><tr><td>не удалось загрузить</td></tr></tbody>";
  }
}

function autosize() {
  els.input.style.height = "auto";
  els.input.style.height = Math.min(els.input.scrollHeight, 160) + "px";
}

els.composer.addEventListener("submit", (e) => { e.preventDefault(); send(els.input.value); });
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(els.input.value); }
});
els.input.addEventListener("input", autosize);
$("starters").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-text]");
  if (btn) send(btn.dataset.text);
});
$("new-session").addEventListener("click", () => {
  if (busy) return;
  storage.clear();
  sessionId = null;
  resetView();
  els.input.focus();
});

// nothing can be sent until the saved dialog is restored, otherwise two sessions mix in one feed
setBusy(true);
restore().finally(() => {
  setBusy(false);
  els.status.dataset.state = els.status.dataset.state === "busy" ? "active" : els.status.dataset.state;
  if (els.status.textContent === "печатает…") els.status.textContent = "агент на связи";
});
loadDemoTrips();
