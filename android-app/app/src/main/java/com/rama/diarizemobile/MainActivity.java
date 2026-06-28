package com.rama.diarizemobile;

import android.Manifest;
import android.app.Activity;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.media.MediaRecorder;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.text.InputType;
import android.view.Gravity;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.EditText;
import android.widget.HorizontalScrollView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.ScrollView;
import android.widget.Space;
import android.widget.ArrayAdapter;
import android.widget.Spinner;
import android.widget.TextView;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends Activity {
    private static final int REQ_RECORD_AUDIO = 10;
    private static final int GREEN = Color.rgb(15, 118, 110);
    private static final int BG = Color.rgb(244, 247, 246);
    private static final int CARD = Color.WHITE;
    private static final int TEXT = Color.rgb(23, 33, 31);
    private static final int MUTED = Color.rgb(96, 113, 109);
    private static final int DANGER = Color.rgb(180, 35, 24);

    private final ExecutorService io = Executors.newSingleThreadExecutor();
    private final Handler main = new Handler(Looper.getMainLooper());

    private SharedPreferences prefs;
    private EditText serverUrlInput;
    private EditText profileInput;
    private Spinner modelInput;
    private Spinner languageInput;
    private Spinner speakerCountInput;
    private Spinner diarizationInput;
    private TextView statusText;
    private TextView recordingText;
    private TextView jobText;
    private ProgressBar jobProgress;
    private LinearLayout lessonsList;
    private LinearLayout detailPanel;
    private Button recordButton;
    private Button stopButton;
    private Button uploadButton;

    private MediaRecorder recorder;
    private File recordingFile;
    private boolean isRecording = false;
    private long recordingStartedAt = 0L;
    private String lastLessonId;

    private final Runnable recordingTicker = new Runnable() {
        @Override
        public void run() {
            if (!isRecording) {
                return;
            }
            long seconds = Math.max(0, (System.currentTimeMillis() - recordingStartedAt) / 1000);
            recordingText.setText(String.format(Locale.US, "Recording %02d:%02d", seconds / 60, seconds % 60));
            main.postDelayed(this, 500);
        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        prefs = getSharedPreferences("diarize", MODE_PRIVATE);
        buildUi();
        setServerStatus("Ready. Connect to the server.");
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (isRecording) {
            stopRecording();
        }
        io.shutdownNow();
    }

    private void buildUi() {
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setBackgroundColor(BG);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setPadding(dp(16), dp(18), dp(16), dp(28));
        scroll.addView(root);

        TextView title = label("Diarize", 30, TEXT, true);
        root.addView(title);
        root.addView(label("Native Android recorder and lesson client", 14, MUTED, false));

        LinearLayout connection = card();
        connection.addView(sectionTitle("Connection"));
        serverUrlInput = edit(prefs.getString("server_url", "http://192.168.1.160:8000"), "Server URL");
        connection.addView(serverUrlInput);
        Button connect = primaryButton("Connect");
        connect.setOnClickListener(v -> checkServer());
        connection.addView(connect);
        statusText = label("", 14, MUTED, false);
        connection.addView(statusText);
        root.addView(connection);

        LinearLayout newLesson = card();
        newLesson.addView(sectionTitle("New Lesson"));
        profileInput = edit(prefs.getString("profile", "default"), "Profile");
        newLesson.addView(profileInput);

        newLesson.addView(fieldLabel("Model size"));
        modelInput = spinner(
                "Model size",
                new String[]{"large-v3", "turbo", "large-v2", "medium", "small", "base", "tiny"},
                prefs.getString("model", "large-v3")
        );
        newLesson.addView(modelInput);

        newLesson.addView(fieldLabel("Language"));
        languageInput = spinner(
                "Language",
                new String[]{"", "es", "en", "fr", "de", "it", "pt", "auto"},
                prefs.getString("language", "es")
        );
        newLesson.addView(languageInput);

        newLesson.addView(fieldLabel("Known speakers"));
        speakerCountInput = spinner(
                "Known speakers",
                new String[]{"", "1", "2", "3", "4", "5", "6", "7", "8"},
                prefs.getString("speaker_count", "2")
        );
        newLesson.addView(speakerCountInput);

        newLesson.addView(fieldLabel("Diarization"));
        diarizationInput = spinner(
                "Diarization",
                new String[]{"pyannote", "single_speaker"},
                prefs.getString("diarization", "pyannote")
        );
        newLesson.addView(diarizationInput);

        LinearLayout buttonRow = new LinearLayout(this);
        buttonRow.setOrientation(LinearLayout.HORIZONTAL);
        buttonRow.setGravity(Gravity.CENTER_VERTICAL);
        recordButton = secondaryButton("Record");
        stopButton = dangerButton("Stop");
        stopButton.setEnabled(false);
        recordButton.setOnClickListener(v -> ensureRecordingPermissionAndStart());
        stopButton.setOnClickListener(v -> stopRecording());
        buttonRow.addView(recordButton, weightParams(1));
        buttonRow.addView(space(dp(8), 1));
        buttonRow.addView(stopButton, weightParams(1));
        newLesson.addView(buttonRow);

        recordingText = label("No recording yet", 14, MUTED, false);
        newLesson.addView(recordingText);

        uploadButton = primaryButton("Upload Recording");
        uploadButton.setEnabled(false);
        uploadButton.setOnClickListener(v -> uploadRecording());
        newLesson.addView(uploadButton);
        root.addView(newLesson);

        LinearLayout job = card();
        job.addView(sectionTitle("Processing"));
        jobText = label("No active job", 14, MUTED, false);
        job.addView(jobText);
        jobProgress = new ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal);
        jobProgress.setMax(100);
        jobProgress.setProgress(0);
        job.addView(jobProgress, matchWrap());
        Button openLast = secondaryButton("Open Last Processed Lesson");
        openLast.setOnClickListener(v -> {
            if (lastLessonId != null) {
                openLesson(lastLessonId);
            }
        });
        job.addView(openLast);
        root.addView(job);

        LinearLayout history = card();
        history.addView(sectionTitle("History"));
        Button refresh = secondaryButton("Refresh Lessons");
        refresh.setOnClickListener(v -> loadLessons());
        history.addView(refresh);
        lessonsList = new LinearLayout(this);
        lessonsList.setOrientation(LinearLayout.VERTICAL);
        history.addView(lessonsList);
        root.addView(history);

        detailPanel = card();
        detailPanel.addView(sectionTitle("Lesson"));
        detailPanel.addView(label("Select a lesson to view transcript, speakers, and analysis.", 14, MUTED, false));
        root.addView(detailPanel);

        setContentView(scroll);
    }

    private void checkServer() {
        persistSettings();
        setServerStatus("Checking server...");
        io.execute(() -> {
            try {
                JSONObject health = requestJson("GET", "/api/health", null);
                runMain(() -> {
                    setServerStatus("Connected. Data: " + health.optString("data_dir", "server storage"));
                    loadLessons();
                });
            } catch (Exception e) {
                runMain(() -> setServerStatus("Connection failed: " + e.getMessage()));
            }
        });
    }

    private void ensureRecordingPermissionAndStart() {
        if (checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.RECORD_AUDIO}, REQ_RECORD_AUDIO);
            return;
        }
        startRecording();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQ_RECORD_AUDIO && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startRecording();
        } else {
            recordingText.setText("Microphone permission is required to record.");
        }
    }

    private void startRecording() {
        try {
            recordingFile = new File(getCacheDir(), "lesson-" + System.currentTimeMillis() + ".m4a");
            recorder = new MediaRecorder();
            recorder.setAudioSource(MediaRecorder.AudioSource.MIC);
            recorder.setOutputFormat(MediaRecorder.OutputFormat.MPEG_4);
            recorder.setAudioEncoder(MediaRecorder.AudioEncoder.AAC);
            recorder.setAudioEncodingBitRate(128000);
            recorder.setAudioSamplingRate(44100);
            recorder.setOutputFile(recordingFile.getAbsolutePath());
            recorder.prepare();
            recorder.start();
            isRecording = true;
            recordingStartedAt = System.currentTimeMillis();
            recordButton.setEnabled(false);
            stopButton.setEnabled(true);
            uploadButton.setEnabled(false);
            main.post(recordingTicker);
        } catch (Exception e) {
            cleanupRecorder();
            recordingText.setText("Recording failed: " + e.getMessage());
        }
    }

    private void stopRecording() {
        try {
            if (recorder != null) {
                recorder.stop();
            }
            recordingText.setText("Saved recording: " + recordingFile.getName());
            uploadButton.setEnabled(recordingFile != null && recordingFile.isFile());
        } catch (RuntimeException e) {
            recordingText.setText("Recording was too short or failed.");
            if (recordingFile != null) {
                recordingFile.delete();
            }
            recordingFile = null;
        } finally {
            cleanupRecorder();
            recordButton.setEnabled(true);
            stopButton.setEnabled(false);
        }
    }

    private void cleanupRecorder() {
        isRecording = false;
        main.removeCallbacks(recordingTicker);
        if (recorder != null) {
            recorder.release();
            recorder = null;
        }
    }

    private void uploadRecording() {
        if (recordingFile == null || !recordingFile.isFile()) {
            recordingText.setText("Record a lesson first.");
            return;
        }
        persistSettings();
        uploadButton.setEnabled(false);
        setJob("Uploading...", 0);
        io.execute(() -> {
            try {
                JSONObject created = postMultipartJob(recordingFile);
                String jobId = created.getString("id");
                runMain(() -> setJob("Queued: " + jobId, 0));
                pollJob(jobId);
            } catch (Exception e) {
                runMain(() -> {
                    uploadButton.setEnabled(true);
                    setJob("Upload failed: " + e.getMessage(), 0);
                });
            }
        });
    }

    private void pollJob(String jobId) throws Exception {
        int consecutiveFailures = 0;
        int lastProgress = 0;
        while (!Thread.currentThread().isInterrupted()) {
            try {
                JSONObject job = requestJson("GET", "/api/jobs/" + jobId, null);
                consecutiveFailures = 0;
                String status = job.optString("status", "unknown");
                String message = job.optString("message", status);
                int progress = (int) Math.round(job.optDouble("progress", 0));
                lastProgress = progress;
                runMain(() -> setJob(message + " (" + status + ")", progress));

                if ("succeeded".equals(status)) {
                    lastLessonId = job.optString("lesson_id", null);
                    runMain(() -> {
                        uploadButton.setEnabled(true);
                        setJob("Done", 100);
                        loadLessons();
                        if (lastLessonId != null && !lastLessonId.isEmpty()) {
                            openLesson(lastLessonId);
                        }
                    });
                    return;
                }
                if ("failed".equals(status)) {
                    String error = job.optString("error", "Processing failed");
                    int failedProgress = progress;
                    runMain(() -> {
                        uploadButton.setEnabled(true);
                        setJob(error, failedProgress);
                    });
                    return;
                }
            } catch (Exception e) {
                consecutiveFailures += 1;
                int failedPolls = consecutiveFailures;
                int retainedProgress = lastProgress;
                runMain(() -> setJob("Connection interrupted; retrying (" + failedPolls + "/12): " + e.getMessage(), retainedProgress));
                if (consecutiveFailures >= 12) {
                    runMain(() -> {
                        uploadButton.setEnabled(true);
                        setJob("Could not reach server after repeated retries: " + e.getMessage(), retainedProgress);
                    });
                    return;
                }
            }
            Thread.sleep(1600);
        }
    }

    private void loadLessons() {
        setServerStatus("Loading lessons...");
        io.execute(() -> {
            try {
                JSONObject data = requestJson("GET", "/api/lessons", null);
                JSONArray lessons = data.optJSONArray("lessons");
                runMain(() -> renderLessons(lessons == null ? new JSONArray() : lessons));
            } catch (Exception e) {
                runMain(() -> setServerStatus("Could not load lessons: " + e.getMessage()));
            }
        });
    }

    private void renderLessons(JSONArray lessons) {
        lessonsList.removeAllViews();
        setServerStatus("Loaded " + lessons.length() + " lesson" + (lessons.length() == 1 ? "" : "s"));
        if (lessons.length() == 0) {
            lessonsList.addView(label("No lessons yet.", 14, MUTED, false));
            return;
        }
        for (int i = 0; i < lessons.length(); i++) {
            JSONObject lesson = lessons.optJSONObject(i);
            if (lesson == null) {
                continue;
            }
            String id = lesson.optString("id");
            String profile = lesson.optString("profile", id);
            String meta = formatDuration(lesson.optDouble("duration_sec", 0)) + " - " + lesson.optInt("num_speakers", 0) + " speakers";
            Button button = secondaryButton(profile + "\n" + meta);
            button.setGravity(Gravity.START | Gravity.CENTER_VERTICAL);
            button.setOnClickListener(v -> openLesson(id));
            lessonsList.addView(button);
        }
    }

    private void openLesson(String lessonId) {
        detailPanel.removeAllViews();
        detailPanel.addView(sectionTitle("Lesson"));
        detailPanel.addView(label("Loading...", 14, MUTED, false));
        io.execute(() -> {
            try {
                JSONObject lesson = requestJson("GET", "/api/lessons/" + lessonId, null);
                runMain(() -> renderLesson(lesson));
            } catch (Exception e) {
                runMain(() -> {
                    detailPanel.removeAllViews();
                    detailPanel.addView(sectionTitle("Lesson"));
                    detailPanel.addView(label("Could not load lesson: " + e.getMessage(), 14, DANGER, false));
                });
            }
        });
    }

    private void renderLesson(JSONObject lesson) {
        detailPanel.removeAllViews();
        JSONObject meta = lesson.optJSONObject("meta");
        JSONArray segments = lesson.optJSONArray("segments");
        if (meta == null) {
            meta = new JSONObject();
        }
        if (segments == null) {
            segments = new JSONArray();
        }

        String id = lesson.optString("id");
        detailPanel.addView(sectionTitle(meta.optString("profile", "Lesson")));
        detailPanel.addView(label(id, 13, MUTED, false));

        HorizontalScrollView actionsScroll = new HorizontalScrollView(this);
        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        Button transcript = secondaryButton("Transcript");
        Button speakers = secondaryButton("Speakers");
        Button analysis = secondaryButton("Analysis");
        actions.addView(transcript);
        actions.addView(space(dp(8), 1));
        actions.addView(speakers);
        actions.addView(space(dp(8), 1));
        actions.addView(analysis);
        actionsScroll.addView(actions);
        detailPanel.addView(actionsScroll);

        LinearLayout content = new LinearLayout(this);
        content.setOrientation(LinearLayout.VERTICAL);
        detailPanel.addView(content);

        JSONObject finalMeta = meta;
        JSONArray finalSegments = segments;
        transcript.setOnClickListener(v -> renderTranscript(content, finalSegments));
        speakers.setOnClickListener(v -> renderSpeakers(content, lesson, finalSegments, finalMeta));
        analysis.setOnClickListener(v -> renderAnalysis(content, lesson));
        renderTranscript(content, finalSegments);
    }

    private void renderTranscript(LinearLayout content, JSONArray segments) {
        content.removeAllViews();
        if (segments.length() == 0) {
            content.addView(label("No transcript segments available.", 14, MUTED, false));
            return;
        }
        for (int i = 0; i < segments.length(); i++) {
            JSONObject segment = segments.optJSONObject(i);
            if (segment == null) {
                continue;
            }
            TextView speaker = label(segment.optString("speaker", "UNKNOWN") + " - " + formatTime(segment.optDouble("start", 0)), 12, GREEN, true);
            TextView text = label(segment.optString("text", ""), 16, TEXT, false);
            LinearLayout block = miniCard();
            block.addView(speaker);
            block.addView(text);
            content.addView(block);
        }
    }

    private void renderSpeakers(LinearLayout content, JSONObject lesson, JSONArray segments, JSONObject meta) {
        content.removeAllViews();
        Set<String> ids = speakerIds(segments);
        if (ids.isEmpty()) {
            content.addView(label("No speakers available.", 14, MUTED, false));
            return;
        }

        JSONObject labels = meta.optJSONObject("speaker_labels");
        JSONArray studentArray = meta.optJSONArray("student_speakers");
        Set<String> students = new LinkedHashSet<>();
        if (studentArray != null) {
            for (int i = 0; i < studentArray.length(); i++) {
                students.add(studentArray.optString(i));
            }
        }

        Map<String, CheckBox> checks = new LinkedHashMap<>();
        Map<String, EditText> edits = new LinkedHashMap<>();
        for (String id : ids) {
            LinearLayout row = miniCard();
            CheckBox check = new CheckBox(this);
            check.setText("Student: " + id);
            check.setTextColor(TEXT);
            check.setChecked(students.contains(id));
            EditText edit = edit(labels == null ? "" : labels.optString(id, ""), "Label");
            row.addView(check);
            row.addView(edit);
            checks.put(id, check);
            edits.put(id, edit);
            content.addView(row);
        }

        Button save = primaryButton("Save Speaker Review");
        save.setOnClickListener(v -> saveSpeakers(lesson.optString("id"), checks, edits));
        content.addView(save);
    }

    private void saveSpeakers(String lessonId, Map<String, CheckBox> checks, Map<String, EditText> edits) {
        try {
            JSONObject payload = new JSONObject();
            JSONObject labels = new JSONObject();
            JSONArray students = new JSONArray();
            for (String id : checks.keySet()) {
                if (checks.get(id).isChecked()) {
                    students.put(id);
                }
                String label = edits.get(id).getText().toString().trim();
                if (!label.isEmpty()) {
                    labels.put(id, label);
                }
            }
            payload.put("speaker_labels", labels);
            payload.put("student_speakers", students);
            detailPanel.addView(label("Saving speaker review...", 14, MUTED, false));
            io.execute(() -> {
                try {
                    JSONObject updated = requestJson("PATCH", "/api/lessons/" + lessonId + "/speakers", payload.toString());
                    runMain(() -> renderLesson(updated));
                } catch (Exception e) {
                    runMain(() -> detailPanel.addView(label("Save failed: " + e.getMessage(), 14, DANGER, false)));
                }
            });
        } catch (Exception e) {
            detailPanel.addView(label("Save failed: " + e.getMessage(), 14, DANGER, false));
        }
    }

    private void renderAnalysis(LinearLayout content, JSONObject lesson) {
        content.removeAllViews();
        JSONObject aiStats = lesson.optJSONObject("ai_stats");
        content.addView(label(aiStats == null ? "No AI metrics yet." : prettyJson(aiStats), 14, TEXT, false));

        EditText provider = edit("ollama", "Provider");
        EditText model = edit("gemma4:e4b", "Model");
        content.addView(provider);
        content.addView(model);

        Button analyze = primaryButton("Analyze Lesson");
        analyze.setOnClickListener(v -> {
            try {
                JSONObject payload = new JSONObject();
                payload.put("provider", provider.getText().toString().trim());
                payload.put("model", model.getText().toString().trim());
                content.addView(label("Analyzing...", 14, MUTED, false));
                io.execute(() -> {
                    try {
                        JSONObject updated = requestJson("POST", "/api/lessons/" + lesson.optString("id") + "/analyze", payload.toString());
                        runMain(() -> renderLesson(updated));
                    } catch (Exception e) {
                        runMain(() -> content.addView(label("Analysis failed: " + e.getMessage(), 14, DANGER, false)));
                    }
                });
            } catch (Exception e) {
                content.addView(label("Analysis failed: " + e.getMessage(), 14, DANGER, false));
            }
        });
        content.addView(analyze);
    }

    private JSONObject postMultipartJob(File audio) throws Exception {
        String boundary = "DiarizeBoundary" + System.currentTimeMillis();
        URL url = new URL(serverUrl() + "/api/jobs");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setDoOutput(true);
        conn.setConnectTimeout(15000);
        conn.setReadTimeout(60000);
        conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);

        try (OutputStream out = new BufferedOutputStream(conn.getOutputStream())) {
            writeFormField(out, boundary, "profile", profileInput.getText().toString().trim());
            writeFormField(out, boundary, "model_size", selectedValue(modelInput));
            String language = selectedValue(languageInput);
            if (!language.isEmpty() && !"auto".equals(language)) {
                writeFormField(out, boundary, "language", language);
            }
            String speakerCount = selectedValue(speakerCountInput);
            if (!speakerCount.isEmpty()) {
                writeFormField(out, boundary, "num_speakers", speakerCount);
            }
            writeFormField(out, boundary, "backend", "auto");
            writeFormField(out, boundary, "diarization_backend", selectedValue(diarizationInput));
            writeFileField(out, boundary, "audio", audio, "audio/mp4");
            out.write(("--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));
        }
        return readJsonResponse(conn);
    }

    private JSONObject requestJson(String method, String path, String body) throws Exception {
        URL url = new URL(serverUrl() + path);
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod(method);
        conn.setConnectTimeout(15000);
        conn.setReadTimeout(60000);
        if (body != null) {
            conn.setDoOutput(true);
            conn.setRequestProperty("Content-Type", "application/json");
            try (OutputStream out = conn.getOutputStream()) {
                out.write(body.getBytes(StandardCharsets.UTF_8));
            }
        }
        return readJsonResponse(conn);
    }

    private JSONObject readJsonResponse(HttpURLConnection conn) throws Exception {
        int code = conn.getResponseCode();
        BufferedInputStream in = new BufferedInputStream(code >= 400 && conn.getErrorStream() != null ? conn.getErrorStream() : conn.getInputStream());
        String text = readAll(in);
        if (code >= 400) {
            String detail = text;
            try {
                detail = new JSONObject(text).optString("detail", text);
            } catch (Exception ignored) {
            }
            throw new IOException(detail);
        }
        return new JSONObject(text);
    }

    private void writeFormField(OutputStream out, String boundary, String name, String value) throws IOException {
        out.write(("--" + boundary + "\r\n").getBytes(StandardCharsets.UTF_8));
        out.write(("Content-Disposition: form-data; name=\"" + name + "\"\r\n\r\n").getBytes(StandardCharsets.UTF_8));
        out.write((value + "\r\n").getBytes(StandardCharsets.UTF_8));
    }

    private void writeFileField(OutputStream out, String boundary, String name, File file, String contentType) throws IOException {
        out.write(("--" + boundary + "\r\n").getBytes(StandardCharsets.UTF_8));
        out.write(("Content-Disposition: form-data; name=\"" + name + "\"; filename=\"" + file.getName() + "\"\r\n").getBytes(StandardCharsets.UTF_8));
        out.write(("Content-Type: " + contentType + "\r\n\r\n").getBytes(StandardCharsets.UTF_8));
        try (FileInputStream input = new FileInputStream(file)) {
            byte[] buf = new byte[8192];
            int n;
            while ((n = input.read(buf)) != -1) {
                out.write(buf, 0, n);
            }
        }
        out.write("\r\n".getBytes(StandardCharsets.UTF_8));
    }

    private String readAll(BufferedInputStream in) throws IOException {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = in.read(buf)) != -1) {
            out.write(buf, 0, n);
        }
        return out.toString("UTF-8");
    }

    private String serverUrl() {
        String url = serverUrlInput.getText().toString().trim();
        while (url.endsWith("/")) {
            url = url.substring(0, url.length() - 1);
        }
        return url;
    }

    private void persistSettings() {
        prefs.edit()
                .putString("server_url", serverUrl())
                .putString("profile", profileInput.getText().toString().trim())
                .putString("model", selectedValue(modelInput))
                .putString("language", selectedValue(languageInput))
                .putString("speaker_count", selectedValue(speakerCountInput))
                .putString("diarization", selectedValue(diarizationInput))
                .apply();
    }

    private Set<String> speakerIds(JSONArray segments) {
        Set<String> ids = new LinkedHashSet<>();
        for (int i = 0; i < segments.length(); i++) {
            JSONObject segment = segments.optJSONObject(i);
            if (segment != null && segment.optString("speaker", null) != null) {
                ids.add(segment.optString("speaker"));
            }
        }
        return ids;
    }

    private TextView label(String text, int sp, int color, boolean bold) {
        TextView view = new TextView(this);
        view.setText(text);
        view.setTextSize(sp);
        view.setTextColor(color);
        view.setPadding(0, dp(4), 0, dp(4));
        if (bold) {
            view.setTypeface(view.getTypeface(), android.graphics.Typeface.BOLD);
        }
        return view;
    }

    private TextView sectionTitle(String text) {
        return label(text, 20, TEXT, true);
    }

    private EditText edit(String value, String hint) {
        EditText edit = new EditText(this);
        edit.setText(value);
        edit.setHint(hint);
        edit.setSingleLine(true);
        edit.setInputType(InputType.TYPE_CLASS_TEXT);
        edit.setPadding(dp(10), 0, dp(10), 0);
        edit.setTextColor(TEXT);
        edit.setHintTextColor(MUTED);
        edit.setLayoutParams(matchWrapWithMargins(0, dp(8), 0, dp(8)));
        return edit;
    }

    private Spinner spinner(String label, String[] values, String selected) {
        Spinner spinner = new Spinner(this);
        ArrayAdapter<String> adapter = new ArrayAdapter<String>(this, android.R.layout.simple_spinner_item, values) {
            @Override
            public android.view.View getView(int position, android.view.View convertView, android.view.ViewGroup parent) {
                TextView view = (TextView) super.getView(position, convertView, parent);
                view.setText(displayValue(values[position]));
                view.setTextColor(TEXT);
                return view;
            }

            @Override
            public android.view.View getDropDownView(int position, android.view.View convertView, android.view.ViewGroup parent) {
                TextView view = (TextView) super.getDropDownView(position, convertView, parent);
                view.setText(displayValue(values[position]));
                view.setTextColor(TEXT);
                view.setPadding(dp(12), dp(10), dp(12), dp(10));
                return view;
            }
        };
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);
        spinner.setPadding(dp(6), 0, dp(6), 0);
        spinner.setLayoutParams(matchWrapWithMargins(0, dp(2), 0, dp(8)));
        int index = 0;
        for (int i = 0; i < values.length; i++) {
            if (values[i].equals(selected)) {
                index = i;
                break;
            }
        }
        spinner.setSelection(index);
        return spinner;
    }

    private TextView fieldLabel(String text) {
        TextView view = label(text, 13, MUTED, false);
        view.setPadding(0, dp(8), 0, 0);
        return view;
    }

    private String selectedValue(Spinner spinner) {
        Object value = spinner.getSelectedItem();
        return value == null ? "" : value.toString().trim();
    }

    private String displayValue(String value) {
        if (value == null || value.isEmpty()) {
            return "Auto";
        }
        if ("es".equals(value)) {
            return "Spanish (es)";
        }
        if ("en".equals(value)) {
            return "English (en)";
        }
        if ("fr".equals(value)) {
            return "French (fr)";
        }
        if ("de".equals(value)) {
            return "German (de)";
        }
        if ("it".equals(value)) {
            return "Italian (it)";
        }
        if ("pt".equals(value)) {
            return "Portuguese (pt)";
        }
        if ("pyannote".equals(value)) {
            return "Speaker diarization";
        }
        if ("single_speaker".equals(value)) {
            return "Single speaker";
        }
        return value;
    }

    private LinearLayout card() {
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(dp(14), dp(14), dp(14), dp(14));
        layout.setBackgroundColor(CARD);
        layout.setLayoutParams(matchWrapWithMargins(0, dp(14), 0, 0));
        return layout;
    }

    private LinearLayout miniCard() {
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setPadding(dp(10), dp(10), dp(10), dp(10));
        layout.setBackgroundColor(Color.rgb(248, 251, 250));
        layout.setLayoutParams(matchWrapWithMargins(0, dp(8), 0, 0));
        return layout;
    }

    private Button primaryButton(String text) {
        Button button = new Button(this);
        button.setText(text);
        button.setTextColor(Color.WHITE);
        button.setBackgroundColor(GREEN);
        button.setAllCaps(false);
        button.setLayoutParams(matchWrapWithMargins(0, dp(8), 0, 0));
        return button;
    }

    private Button secondaryButton(String text) {
        Button button = new Button(this);
        button.setText(text);
        button.setTextColor(Color.rgb(11, 79, 73));
        button.setBackgroundColor(Color.rgb(228, 239, 237));
        button.setAllCaps(false);
        button.setLayoutParams(matchWrapWithMargins(0, dp(8), 0, 0));
        return button;
    }

    private Button dangerButton(String text) {
        Button button = secondaryButton(text);
        button.setTextColor(DANGER);
        button.setBackgroundColor(Color.rgb(252, 232, 229));
        return button;
    }

    private LinearLayout.LayoutParams matchWrap() {
        return new LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
    }

    private LinearLayout.LayoutParams weightParams(float weight) {
        return new LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, weight);
    }

    private LinearLayout.LayoutParams matchWrapWithMargins(int left, int top, int right, int bottom) {
        LinearLayout.LayoutParams params = matchWrap();
        params.setMargins(left, top, right, bottom);
        return params;
    }

    private Space space(int width, int height) {
        Space space = new Space(this);
        space.setLayoutParams(new LinearLayout.LayoutParams(width, height));
        return space;
    }

    private void setServerStatus(String text) {
        statusText.setText(text);
    }

    private void setJob(String text, int progress) {
        jobText.setText(text);
        jobProgress.setProgress(Math.max(0, Math.min(100, progress)));
    }

    private String formatDuration(double seconds) {
        int total = (int) Math.max(0, Math.round(seconds));
        return String.format(Locale.US, "%d:%02d", total / 60, total % 60);
    }

    private String formatTime(double seconds) {
        return formatDuration(seconds);
    }

    private String prettyJson(JSONObject json) {
        try {
            return json.toString(2);
        } catch (Exception e) {
            return json.toString();
        }
    }

    private int dp(int value) {
        return (int) (value * getResources().getDisplayMetrics().density + 0.5f);
    }

    private void runMain(Runnable runnable) {
        main.post(runnable);
    }
}
