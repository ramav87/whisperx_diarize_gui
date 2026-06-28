const state = {
  serverUrl: localStorage.getItem("diarize.serverUrl") || defaultServerUrl(),
  selectedAudio: null,
  selectedLessonId: null,
  lessons: [],
  currentLesson: null,
  mediaRecorder: null,
  recordChunks: [],
  recordStartedAt: null,
  recordTimerId: null,
  pollingId: null,
  pollFailures: 0,
  lastJobLessonId: null,
};

const els = {
  serverForm: document.querySelector("#serverForm"),
  serverUrl: document.querySelector("#serverUrl"),
  connectionStatus: document.querySelector("#connectionStatus"),
  connectionBadge: document.querySelector("#connectionBadge"),
  profileInput: document.querySelector("#profileInput"),
  languageSelect: document.querySelector("#languageSelect"),
  modelSelect: document.querySelector("#modelSelect"),
  speakerCountSelect: document.querySelector("#speakerCountSelect"),
  diarizationSelect: document.querySelector("#diarizationSelect"),
  recordBtn: document.querySelector("#recordBtn"),
  stopBtn: document.querySelector("#stopBtn"),
  recordingTimer: document.querySelector("#recordingTimer"),
  audioInput: document.querySelector("#audioInput"),
  submitJobBtn: document.querySelector("#submitJobBtn"),
  selectedAudio: document.querySelector("#selectedAudio"),
  jobPanel: document.querySelector("#jobPanel"),
  jobMessage: document.querySelector("#jobMessage"),
  jobStatus: document.querySelector("#jobStatus"),
  jobProgress: document.querySelector("#jobProgress"),
  openLessonBtn: document.querySelector("#openLessonBtn"),
  refreshLessonsBtn: document.querySelector("#refreshLessonsBtn"),
  lessonCount: document.querySelector("#lessonCount"),
  lessonsList: document.querySelector("#lessonsList"),
  lessonDetail: document.querySelector("#lessonDetail"),
  lessonDetailTemplate: document.querySelector("#lessonDetailTemplate"),
};

init();

function init() {
  els.serverUrl.value = state.serverUrl;
  els.serverForm.addEventListener("submit", handleServerSubmit);
  els.audioInput.addEventListener("change", handleFileSelect);
  els.recordBtn.addEventListener("click", startRecording);
  els.stopBtn.addEventListener("click", stopRecording);
  els.submitJobBtn.addEventListener("click", submitJob);
  els.refreshLessonsBtn.addEventListener("click", loadLessons);
  els.openLessonBtn.addEventListener("click", () => {
    if (state.lastJobLessonId) {
      openLesson(state.lastJobLessonId);
    }
  });

  checkConnection();
}

function defaultServerUrl() {
  const { protocol, hostname } = window.location;
  if (hostname && hostname !== "localhost" && hostname !== "127.0.0.1") {
    return `${protocol}//${hostname}:8000`;
  }
  return "http://127.0.0.1:8000";
}

async function handleServerSubmit(event) {
  event.preventDefault();
  state.serverUrl = normalizeServerUrl(els.serverUrl.value);
  els.serverUrl.value = state.serverUrl;
  localStorage.setItem("diarize.serverUrl", state.serverUrl);
  await checkConnection();
}

async function checkConnection() {
  setBadge(els.connectionBadge, "neutral", "Checking");
  els.connectionStatus.textContent = "Checking server health...";
  try {
    const health = await apiGet("/api/health");
    setBadge(els.connectionBadge, "ok", "Online");
    els.connectionStatus.textContent = `Connected. Data: ${health.data_dir || "server storage"}`;
    await loadLessons();
  } catch (error) {
    setBadge(els.connectionBadge, "error", "Offline");
    els.connectionStatus.textContent = error.message;
  }
}

function normalizeServerUrl(value) {
  return String(value || "").trim().replace(/\/+$/, "");
}

function handleFileSelect() {
  const file = els.audioInput.files && els.audioInput.files[0];
  if (!file) {
    setSelectedAudio(null);
    return;
  }
  setSelectedAudio(file);
}

function setSelectedAudio(file) {
  state.selectedAudio = file;
  els.submitJobBtn.disabled = !file;
  els.selectedAudio.textContent = file ? `${file.name} (${formatBytes(file.size)})` : "No audio selected";
}

async function startRecording() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showInlineError("Recording is not available in this browser. Use the file picker.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = supportedRecordingMimeType();
    state.recordChunks = [];
    state.mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    state.mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) {
        state.recordChunks.push(event.data);
      }
    });
    state.mediaRecorder.addEventListener("stop", () => finishRecording(stream, mimeType));
    state.mediaRecorder.start();
    state.recordStartedAt = Date.now();
    state.recordTimerId = window.setInterval(updateRecordingTimer, 500);
    els.recordBtn.disabled = true;
    els.stopBtn.disabled = false;
    updateRecordingTimer();
  } catch (error) {
    showInlineError(error.message || "Could not start recording.");
  }
}

function stopRecording() {
  if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
    state.mediaRecorder.stop();
  }
}

function finishRecording(stream, mimeType) {
  stream.getTracks().forEach((track) => track.stop());
  window.clearInterval(state.recordTimerId);
  state.recordTimerId = null;
  els.recordBtn.disabled = false;
  els.stopBtn.disabled = true;

  const type = mimeType || "audio/webm";
  const ext = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm";
  const blob = new Blob(state.recordChunks, { type });
  const file = new File([blob], `lesson-recording-${timestampForFile()}.${ext}`, { type });
  setSelectedAudio(file);
}

function supportedRecordingMimeType() {
  const options = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4", "audio/ogg;codecs=opus"];
  return options.find((type) => window.MediaRecorder && MediaRecorder.isTypeSupported(type)) || "";
}

function updateRecordingTimer() {
  if (!state.recordStartedAt) {
    els.recordingTimer.textContent = "00:00";
    return;
  }
  const elapsed = Math.floor((Date.now() - state.recordStartedAt) / 1000);
  const minutes = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const seconds = String(elapsed % 60).padStart(2, "0");
  els.recordingTimer.textContent = `${minutes}:${seconds}`;
}

async function submitJob() {
  if (!state.selectedAudio) {
    return;
  }

  const form = new FormData();
  form.append("audio", state.selectedAudio, state.selectedAudio.name);
  form.append("profile", els.profileInput.value.trim() || "default");
  form.append("model_size", els.modelSelect.value);
  form.append("backend", "auto");
  form.append("diarization_backend", els.diarizationSelect.value || "pyannote");
  if (els.languageSelect.value) {
    form.append("language", els.languageSelect.value);
  }
  if (els.speakerCountSelect.value) {
    form.append("num_speakers", els.speakerCountSelect.value);
  }

  els.submitJobBtn.disabled = true;
  showJobPanel("queued", "Queued", 0);

  try {
    const job = await apiFetch("/api/jobs", { method: "POST", body: form });
    pollJob(job.id);
  } catch (error) {
    showJobPanel("failed", error.message, 0);
    els.submitJobBtn.disabled = false;
  }
}

function pollJob(jobId) {
  window.clearInterval(state.pollingId);
  state.pollFailures = 0;
  state.pollingId = window.setInterval(async () => {
    try {
      const job = await apiGet(`/api/jobs/${encodeURIComponent(jobId)}`);
      state.pollFailures = 0;
      showJobPanel(job.status, job.message || job.status, job.progress || 0);

      if (job.status === "succeeded") {
        window.clearInterval(state.pollingId);
        state.lastJobLessonId = job.lesson_id || (job.result && job.result.lesson_id);
        els.openLessonBtn.classList.toggle("hidden", !state.lastJobLessonId);
        els.submitJobBtn.disabled = false;
        await loadLessons();
      }

      if (job.status === "failed") {
        window.clearInterval(state.pollingId);
        showJobPanel("failed", job.error || job.message || "Processing failed", job.progress || 0);
        els.submitJobBtn.disabled = false;
      }
    } catch (error) {
      state.pollFailures += 1;
      showJobPanel(
        "running",
        `Connection interrupted; retrying (${state.pollFailures}/12): ${error.message}`,
        Number.parseFloat(els.jobProgress.style.width) || 0,
      );
      if (state.pollFailures >= 12) {
        window.clearInterval(state.pollingId);
        showJobPanel("failed", `Could not reach server after repeated retries: ${error.message}`, 0);
        els.submitJobBtn.disabled = false;
      }
    }
  }, 1600);
}

function showJobPanel(status, message, progress) {
  els.jobPanel.classList.remove("hidden");
  els.jobMessage.textContent = message;
  els.jobProgress.style.width = `${Math.max(0, Math.min(100, Number(progress) || 0))}%`;

  if (status === "succeeded") {
    setBadge(els.jobStatus, "ok", "Done");
  } else if (status === "failed") {
    setBadge(els.jobStatus, "error", "Failed");
  } else if (status === "running" || status === "queued") {
    setBadge(els.jobStatus, "busy", status === "running" ? "Running" : "Queued");
  } else {
    setBadge(els.jobStatus, "neutral", status || "Job");
  }
}

async function loadLessons() {
  try {
    const data = await apiGet("/api/lessons");
    state.lessons = data.lessons || [];
    renderLessons();
  } catch (error) {
    els.lessonCount.textContent = error.message;
  }
}

function renderLessons() {
  els.lessonsList.replaceChildren();
  els.lessonCount.textContent = `${state.lessons.length} lesson${state.lessons.length === 1 ? "" : "s"}`;

  if (!state.lessons.length) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "No lessons yet.";
    els.lessonsList.append(empty);
    return;
  }

  for (const lesson of state.lessons) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `lesson-item${lesson.id === state.selectedLessonId ? " active" : ""}`;
    button.innerHTML = `
      <strong>${escapeHtml(displayLessonTitle(lesson))}</strong>
      <span class="muted">${escapeHtml(compactLessonMeta(lesson))}</span>
    `;
    button.addEventListener("click", () => openLesson(lesson.id));
    els.lessonsList.append(button);
  }
}

async function openLesson(lessonId) {
  state.selectedLessonId = lessonId;
  renderLessons();
  els.lessonDetail.innerHTML = `<div class="empty-state"><h2>Loading</h2><p class="muted">Fetching lesson...</p></div>`;
  try {
    const lesson = await apiGet(`/api/lessons/${encodeURIComponent(lessonId)}`);
    state.currentLesson = lesson;
    renderLessonDetail(lesson);
  } catch (error) {
    els.lessonDetail.innerHTML = `<div class="empty-state"><h2>Could not load lesson</h2><p class="muted">${escapeHtml(error.message)}</p></div>`;
  }
}

function renderLessonDetail(lesson) {
  const fragment = els.lessonDetailTemplate.content.cloneNode(true);
  const root = document.createElement("div");
  root.append(fragment);
  const meta = lesson.meta || {};

  root.querySelector(".lesson-date").textContent = meta.processed_at || lesson.id;
  root.querySelector(".lesson-title").textContent = meta.profile || "Lesson";
  root.querySelector(".lesson-meta").textContent = `${formatDuration(meta.duration_sec)} - ${meta.num_segments || lesson.segments.length || 0} segments - ${meta.num_speakers || speakerIds(lesson).length || 0} speakers`;
  root.querySelector(".export-transcript-button").addEventListener("click", () => exportTranscript(lesson));

  renderTranscript(root.querySelector(".transcript"), lesson);
  renderSpeakerForm(root.querySelector(".speaker-form"), lesson);
  renderAnalysis(root, lesson);
  wireTabs(root);

  els.lessonDetail.replaceChildren(...Array.from(root.childNodes));
}

function renderTranscript(container, lesson) {
  container.replaceChildren();
  const segments = lesson.segments || [];
  if (!segments.length) {
    container.innerHTML = `<p class="muted">No transcript segments available.</p>`;
    return;
  }

  for (const segment of segments) {
    const row = document.createElement("article");
    row.className = "segment";
    const times = segment.start != null && segment.end != null ? ` - ${formatTime(segment.start)}-${formatTime(segment.end)}` : "";
    row.innerHTML = `
      <div class="segment-speaker">${escapeHtml(segment.speaker || "UNKNOWN")}${times}</div>
      <div class="segment-text">${escapeHtml(segment.text || "")}</div>
    `;
    container.append(row);
  }
}

function exportTranscript(lesson) {
  const transcript = lesson.transcript || transcriptFromSegments(lesson);
  if (!transcript.trim()) {
    showInlineError("No transcript text is available to export.");
    return;
  }

  const blob = new Blob([transcript], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `${safeFilename(lesson.id || "lesson")}_transcript.txt`;
  document.body.append(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function transcriptFromSegments(lesson) {
  return (lesson.segments || [])
    .map((segment) => {
      const speaker = segment.speaker || "UNKNOWN";
      const text = segment.text || "";
      return `${speaker}: ${text}`;
    })
    .join("\n");
}

function renderSpeakerForm(form, lesson) {
  const meta = lesson.meta || {};
  const labels = meta.speaker_labels || {};
  const students = new Set(meta.student_speakers || []);
  const speakers = speakerIds(lesson);

  form.replaceChildren();
  if (!speakers.length) {
    form.innerHTML = `<p class="muted">No speakers available for review.</p>`;
    return;
  }

  for (const speaker of speakers) {
    const row = document.createElement("label");
    row.className = "speaker-row";
    row.innerHTML = `
      <input type="checkbox" name="student" value="${escapeHtml(speaker)}" ${students.has(speaker) ? "checked" : ""}>
      <span class="speaker-label-fields">
        <span>${escapeHtml(speaker)}</span>
        <input type="text" data-speaker-label="${escapeHtml(speaker)}" value="${escapeHtml(labels[speaker] || "")}" placeholder="Label, e.g. Student or Tutor">
      </span>
    `;
    form.append(row);
  }

  const save = document.createElement("button");
  save.type = "submit";
  save.className = "primary full-width";
  save.textContent = "Save Speaker Review";
  form.append(save);
  form.addEventListener("submit", (event) => saveSpeakerReview(event, lesson, speakers));
}

async function saveSpeakerReview(event, lesson, speakers) {
  event.preventDefault();
  const form = event.currentTarget;
  const payload = {
    speaker_labels: {},
    student_speakers: Array.from(form.querySelectorAll('input[name="student"]:checked')).map((input) => input.value),
  };

  for (const speaker of speakers) {
    const input = form.querySelector(`[data-speaker-label="${escapeAttribute(speaker)}"]`);
    if (input && input.value.trim()) {
      payload.speaker_labels[speaker] = input.value.trim();
    }
  }

  try {
    const updated = await apiFetch(`/api/lessons/${encodeURIComponent(lesson.id)}/speakers`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    state.currentLesson = updated;
    renderLessonDetail(updated);
  } catch (error) {
    showInlineError(error.message);
  }
}

function renderAnalysis(root, lesson) {
  const output = root.querySelector(".analysis-output");
  output.textContent = lesson.ai_stats ? JSON.stringify(lesson.ai_stats, null, 2) : "No AI metrics yet.";
  root.querySelector(".analyze-button").addEventListener("click", async () => {
    const provider = root.querySelector(".analysis-provider").value;
    const model = root.querySelector(".analysis-model").value.trim();
    output.textContent = "Analyzing...";
    try {
      const updated = await apiFetch(`/api/lessons/${encodeURIComponent(lesson.id)}/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ provider, model }),
      });
      state.currentLesson = updated;
      renderLessonDetail(updated);
    } catch (error) {
      output.textContent = error.message;
    }
  });
}

function wireTabs(root) {
  const tabs = Array.from(root.querySelectorAll(".tab"));
  const panels = Array.from(root.querySelectorAll("[data-view-panel]"));
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const view = tab.dataset.view;
      tabs.forEach((item) => item.classList.toggle("active", item === tab));
      panels.forEach((panel) => panel.classList.toggle("hidden", panel.dataset.viewPanel !== view));
    });
  });
}

async function apiGet(path) {
  return apiFetch(path);
}

async function apiFetch(path, options = {}) {
  const response = await fetch(`${state.serverUrl}${path}`, options);
  let body = null;
  const text = await response.text();
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }
  if (!response.ok) {
    const detail = body && typeof body === "object" ? body.detail : body;
    throw new Error(detail || `Request failed with ${response.status}`);
  }
  return body;
}

function setBadge(element, kind, text) {
  element.className = `status-badge ${kind}`;
  element.textContent = text;
}

function showInlineError(message) {
  setBadge(els.connectionBadge, "error", "Error");
  els.connectionStatus.textContent = message;
}

function speakerIds(lesson) {
  return Array.from(new Set((lesson.segments || []).map((segment) => segment.speaker).filter(Boolean))).sort();
}

function displayLessonTitle(lesson) {
  return lesson.profile || lesson.id;
}

function compactLessonMeta(lesson) {
  const parts = [];
  if (lesson.processed_at) {
    parts.push(formatDate(lesson.processed_at));
  }
  if (lesson.duration_sec) {
    parts.push(formatDuration(lesson.duration_sec));
  }
  if (lesson.num_speakers) {
    parts.push(`${lesson.num_speakers} speakers`);
  }
  return parts.join(" - ") || lesson.id;
}

function formatDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }).format(date);
}

function formatDuration(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds) || seconds <= 0) {
    return "0:00";
  }
  const minutes = Math.floor(seconds / 60);
  const rest = Math.floor(seconds % 60);
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

function formatTime(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) {
    return "0:00";
  }
  const minutes = Math.floor(seconds / 60);
  const rest = Math.floor(seconds % 60);
  return `${minutes}:${String(rest).padStart(2, "0")}`;
}

function formatBytes(bytes) {
  if (!bytes) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
}

function timestampForFile() {
  return new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
}

function safeFilename(value) {
  return String(value || "lesson")
    .trim()
    .replace(/[^A-Za-z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "") || "lesson";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeAttribute(value) {
  return String(value ?? "").replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}
