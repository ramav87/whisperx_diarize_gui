const state = {
  lessons: [],
  selectedLesson: null,
  selectedFile: null,
  mediaRecorder: null,
  chunks: [],
  recordStartedAt: null,
  recordTimerId: null,
  activeJobTimer: null,
  activeAnalysisTimer: null,
};

const el = {
  serverUrl: document.querySelector("#serverUrl"),
  healthBadge: document.querySelector("#healthBadge"),
  refreshLessons: document.querySelector("#refreshLessons"),
  profile: document.querySelector("#profile"),
  language: document.querySelector("#language"),
  modelSize: document.querySelector("#modelSize"),
  numSpeakers: document.querySelector("#numSpeakers"),
  recordButton: document.querySelector("#recordButton"),
  stopButton: document.querySelector("#stopButton"),
  recordTimer: document.querySelector("#recordTimer"),
  fileInput: document.querySelector("#fileInput"),
  uploadButton: document.querySelector("#uploadButton"),
  jobBox: document.querySelector("#jobBox"),
  jobProgress: document.querySelector("#jobProgress"),
  jobMessage: document.querySelector("#jobMessage"),
  lessonsList: document.querySelector("#lessonsList"),
  detailTitle: document.querySelector("#detailTitle"),
  analyzeButton: document.querySelector("#analyzeButton"),
  speakerEditor: document.querySelector("#speakerEditor"),
  statsBox: document.querySelector("#statsBox"),
  segmentsList: document.querySelector("#segmentsList"),
  toast: document.querySelector("#toast"),
};

function apiBase() {
  return el.serverUrl.value.trim().replace(/\/$/, "");
}

function apiUrl(path) {
  return `${apiBase()}${path}`;
}

async function request(path, options = {}) {
  const response = await fetch(apiUrl(path), options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || detail;
    } catch (_err) {
      detail = await response.text();
    }
    throw new Error(detail);
  }
  return response.json();
}

function showToast(message) {
  el.toast.textContent = message;
  el.toast.classList.remove("hidden");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => el.toast.classList.add("hidden"), 3600);
}

function setHealth(ok, text) {
  el.healthBadge.textContent = text;
  el.healthBadge.classList.toggle("bad", !ok);
}

async function checkHealth() {
  try {
    await request("/api/health");
    setHealth(true, "Online");
  } catch (err) {
    setHealth(false, "Offline");
  }
}

async function loadLessons() {
  try {
    const body = await request("/api/lessons");
    state.lessons = body.lessons || [];
    renderLessons();
  } catch (err) {
    showToast(`Could not load lessons: ${err.message}`);
  }
}

function renderLessons() {
  el.lessonsList.innerHTML = "";
  if (!state.lessons.length) {
    el.lessonsList.innerHTML = '<p class="lessonMeta">No lessons yet.</p>';
    return;
  }

  for (const lesson of state.lessons) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "lessonItem";
    if (state.selectedLesson && lesson.id === state.selectedLesson.id) {
      button.classList.add("active");
    }
    button.innerHTML = `
      <strong>${escapeHtml(lesson.profile || "Lesson")}</strong>
      <p class="lessonMeta">${escapeHtml(lesson.id)} · ${formatDuration(lesson.duration_sec)}</p>
      <p class="lessonMeta">${lesson.num_speakers || "?"} speakers · ${lesson.num_segments || 0} segments</p>
    `;
    button.addEventListener("click", () => loadLesson(lesson.id));
    el.lessonsList.appendChild(button);
  }
}

async function loadLesson(id) {
  try {
    state.selectedLesson = await request(`/api/lessons/${encodeURIComponent(id)}`);
    renderLessons();
    renderLessonDetail();
  } catch (err) {
    showToast(`Could not load lesson: ${err.message}`);
  }
}

function renderLessonDetail() {
  const lesson = state.selectedLesson;
  if (!lesson) return;

  el.detailTitle.textContent = lesson.meta.profile || lesson.id;
  el.analyzeButton.disabled = false;
  renderSpeakerEditor(lesson);
  renderStats(lesson.ai_stats);
  renderSegments(lesson);
}

function renderSpeakerEditor(lesson) {
  const speakers = [...new Set((lesson.segments || []).map((seg) => seg.speaker).filter(Boolean))].sort();
  const labels = lesson.meta.speaker_labels || {};
  const students = new Set(lesson.meta.student_speakers || []);

  el.speakerEditor.innerHTML = "";
  if (!speakers.length) {
    el.speakerEditor.classList.add("hidden");
    return;
  }

  for (const speaker of speakers) {
    const row = document.createElement("div");
    row.className = "speakerRow";
    row.dataset.speaker = speaker;
    row.innerHTML = `
      <strong>${escapeHtml(speaker)}</strong>
      <input type="text" value="${escapeAttr(labels[speaker] || "")}" placeholder="Label">
      <label><input type="checkbox" ${students.has(speaker) ? "checked" : ""}> Student</label>
    `;
    el.speakerEditor.appendChild(row);
  }

  const save = document.createElement("button");
  save.type = "button";
  save.className = "primaryButton";
  save.textContent = "Save speakers";
  save.addEventListener("click", saveSpeakers);
  el.speakerEditor.appendChild(save);
  el.speakerEditor.classList.remove("hidden");
}

async function saveSpeakers() {
  if (!state.selectedLesson) return;
  const speaker_labels = {};
  const student_speakers = [];
  for (const row of el.speakerEditor.querySelectorAll(".speakerRow")) {
    const speaker = row.dataset.speaker;
    const label = row.querySelector('input[type="text"]').value.trim();
    const isStudent = row.querySelector('input[type="checkbox"]').checked;
    if (label) speaker_labels[speaker] = label;
    if (isStudent) student_speakers.push(speaker);
  }

  try {
    state.selectedLesson = await request(`/api/lessons/${encodeURIComponent(state.selectedLesson.id)}/speakers`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ speaker_labels, student_speakers }),
    });
    showToast("Speaker review saved");
    renderLessonDetail();
  } catch (err) {
    showToast(`Could not save speakers: ${err.message}`);
  }
}

function renderStats(stats) {
  if (!stats) {
    el.statsBox.classList.add("hidden");
    el.statsBox.innerHTML = "";
    return;
  }
  el.statsBox.innerHTML = `
    ${statItem("Grammar", stats.grammar_score ?? "-")}
    ${statItem("Fluency", Math.round(stats.context_metrics?.effective_fluency_score ?? 0) || "-")}
    ${statItem("Complexity", Math.round(stats.context_metrics?.complexity_resilience_score ?? 0) || "-")}
  `;
  el.statsBox.classList.remove("hidden");
}

function renderSegments(lesson) {
  const labels = lesson.meta.speaker_labels || {};
  el.segmentsList.innerHTML = "";
  for (const seg of lesson.segments || []) {
    const item = document.createElement("article");
    item.className = "segmentItem";
    const label = labels[seg.speaker] ? `${labels[seg.speaker]} (${seg.speaker})` : seg.speaker;
    item.innerHTML = `
      <p class="segmentSpeaker">${escapeHtml(label || "Unknown")} · ${formatTime(seg.start)}-${formatTime(seg.end)}</p>
      <p class="segmentText">${escapeHtml(seg.text || "")}</p>
    `;
    el.segmentsList.appendChild(item);
  }
}

async function startRecording() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    state.chunks = [];
    state.mediaRecorder = new MediaRecorder(stream);
    state.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size) state.chunks.push(event.data);
    };
    state.mediaRecorder.onstop = () => {
      stream.getTracks().forEach((track) => track.stop());
      const blob = new Blob(state.chunks, { type: state.mediaRecorder.mimeType || "audio/webm" });
      state.selectedFile = new File([blob], `lesson-${Date.now()}.webm`, { type: blob.type });
      el.uploadButton.disabled = false;
      stopRecordTimer();
    };
    state.mediaRecorder.start();
    state.recordStartedAt = Date.now();
    state.recordTimerId = window.setInterval(updateRecordTimer, 500);
    el.recordButton.disabled = true;
    el.stopButton.disabled = false;
  } catch (err) {
    showToast(`Recording unavailable: ${err.message}`);
  }
}

function stopRecording() {
  if (state.mediaRecorder && state.mediaRecorder.state !== "inactive") {
    state.mediaRecorder.stop();
  }
  el.recordButton.disabled = false;
  el.stopButton.disabled = true;
}

function updateRecordTimer() {
  const seconds = Math.floor((Date.now() - state.recordStartedAt) / 1000);
  el.recordTimer.textContent = `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

function stopRecordTimer() {
  window.clearInterval(state.recordTimerId);
  state.recordTimerId = null;
}

async function uploadSelected() {
  if (!state.selectedFile) return;
  const form = new FormData();
  form.append("audio", state.selectedFile);
  form.append("profile", el.profile.value.trim() || "default");
  form.append("model_size", el.modelSize.value);
  if (el.language.value.trim()) form.append("language", el.language.value.trim());
  if (el.numSpeakers.value) form.append("num_speakers", el.numSpeakers.value);

  try {
    const job = await request("/api/jobs", { method: "POST", body: form });
    showJob(job);
    pollProcessingJob(job.id);
  } catch (err) {
    showToast(`Upload failed: ${err.message}`);
  }
}

function showJob(job) {
  el.jobBox.classList.remove("hidden");
  el.jobProgress.style.width = `${Math.max(0, Math.min(100, job.progress || 0))}%`;
  el.jobMessage.textContent = job.message || job.status;
}

function pollProcessingJob(jobId) {
  window.clearInterval(state.activeJobTimer);
  state.activeJobTimer = window.setInterval(async () => {
    try {
      const job = await request(`/api/jobs/${jobId}`);
      showJob(job);
      if (job.status === "succeeded") {
        window.clearInterval(state.activeJobTimer);
        await loadLessons();
        await loadLesson(job.result.lesson_id || job.lesson_id);
      } else if (job.status === "failed") {
        window.clearInterval(state.activeJobTimer);
        showToast(job.error || "Processing failed");
      }
    } catch (err) {
      window.clearInterval(state.activeJobTimer);
      showToast(`Could not poll job: ${err.message}`);
    }
  }, 1800);
}

async function startAnalysis() {
  if (!state.selectedLesson) return;
  try {
    const job = await request(`/api/lessons/${encodeURIComponent(state.selectedLesson.id)}/analysis-jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider: "ollama", model: "gemma4:e4b" }),
    });
    el.analyzeButton.disabled = true;
    el.analyzeButton.textContent = "Analyzing";
    pollAnalysisJob(job.id);
  } catch (err) {
    showToast(`Could not start analysis: ${err.message}`);
  }
}

function pollAnalysisJob(jobId) {
  window.clearInterval(state.activeAnalysisTimer);
  state.activeAnalysisTimer = window.setInterval(async () => {
    try {
      const job = await request(`/api/analysis-jobs/${jobId}`);
      if (job.status === "succeeded") {
        window.clearInterval(state.activeAnalysisTimer);
        el.analyzeButton.textContent = "Analyze";
        el.analyzeButton.disabled = false;
        state.selectedLesson = job.result;
        renderLessonDetail();
      } else if (job.status === "failed") {
        window.clearInterval(state.activeAnalysisTimer);
        el.analyzeButton.textContent = "Analyze";
        el.analyzeButton.disabled = false;
        showToast(job.error || "Analysis failed");
      }
    } catch (err) {
      window.clearInterval(state.activeAnalysisTimer);
      showToast(`Could not poll analysis: ${err.message}`);
    }
  }, 1800);
}

function statItem(label, value) {
  return `<div class="statItem"><p class="statLabel">${escapeHtml(label)}</p><p class="statValue">${escapeHtml(String(value))}</p></div>`;
}

function formatDuration(seconds) {
  if (!seconds) return "unknown length";
  const min = Math.floor(seconds / 60);
  const sec = Math.round(seconds % 60);
  return `${min}m ${sec}s`;
}

function formatTime(seconds) {
  if (seconds == null) return "--:--";
  const value = Number(seconds);
  return `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, "0")}`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  })[char]);
}

function escapeAttr(value) {
  return escapeHtml(value);
}

el.refreshLessons.addEventListener("click", loadLessons);
el.fileInput.addEventListener("change", () => {
  state.selectedFile = el.fileInput.files[0] || null;
  el.uploadButton.disabled = !state.selectedFile;
});
el.recordButton.addEventListener("click", startRecording);
el.stopButton.addEventListener("click", stopRecording);
el.uploadButton.addEventListener("click", uploadSelected);
el.analyzeButton.addEventListener("click", startAnalysis);
el.serverUrl.addEventListener("change", () => {
  checkHealth();
  loadLessons();
});

checkHealth();
loadLessons();
