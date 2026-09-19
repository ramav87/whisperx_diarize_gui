package com.rama.diarizemobile;

import android.Manifest;
import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.graphics.Color;
import android.media.MediaRecorder;
import android.net.Uri;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.provider.OpenableColumns;
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
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.net.URLEncoder;
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
    private static final int REQ_PICK_AUDIO = 11;
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
    private Spinner backendInput;
    private Spinner diarizationInput;
    private Spinner batchSizeInput;
    private TextView statusText;
    private TextView recordingText;
    private TextView jobText;
    private ProgressBar jobProgress;
    private LinearLayout lessonsList;
    private LinearLayout detailPanel;
    private Button recordButton;
    private Button stopButton;
    private Button uploadButton;
    private Button pickAudioButton;

    private MediaRecorder recorder;
    private File recordingFile;
    private File selectedAudioFile;
    private String selectedAudioMime = "audio/mp4";
    private boolean isRecording = false;
    private long recordingStartedAt = 0L;
    private String lastLessonId;
    private String openedLessonId;

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

        newLesson.addView(fieldLabel("ASR backend"));
        backendInput = spinner(
                "ASR backend",
                new String[]{"auto", "mlx", "whisper_mps", "whisperx"},
                prefs.getString("backend", "auto")
        );
        newLesson.addView(backendInput);

        newLesson.addView(fieldLabel("Diarization"));
        diarizationInput = spinner(
                "Diarization",
                new String[]{"pyannote", "single_speaker"},
                prefs.getString("diarization", "pyannote")
        );
        newLesson.addView(diarizationInput);

        newLesson.addView(fieldLabel("Batch size"));
        batchSizeInput = spinner(
                "Batch size",
                new String[]{"", "1", "2", "4", "8"},
                prefs.getString("batch_size", "2")
        );
        newLesson.addView(batchSizeInput);

        LinearLayout buttonRow = new LinearLayout(this);
        buttonRow.setOrientation(LinearLayout.HORIZONTAL);
        buttonRow.setGravity(Gravity.CENTER_VERTICAL);
        pickAudioButton = secondaryButton("Pick Audio");
        recordButton = secondaryButton("Record");
        stopButton = dangerButton("Stop");
        stopButton.setEnabled(false);
        pickAudioButton.setOnClickListener(v -> pickAudioFile());
        recordButton.setOnClickListener(v -> ensureRecordingPermissionAndStart());
        stopButton.setOnClickListener(v -> stopRecording());
        buttonRow.addView(pickAudioButton, weightParams(1));
        buttonRow.addView(space(dp(8), 1));
        buttonRow.addView(recordButton, weightParams(1));
        buttonRow.addView(space(dp(8), 1));
        buttonRow.addView(stopButton, weightParams(1));
        newLesson.addView(buttonRow);

        recordingText = label("No recording yet", 14, MUTED, false);
        newLesson.addView(recordingText);

        uploadButton = primaryButton("Upload Audio");
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
        Button recoverJobs = secondaryButton("Recover Server Jobs");
        recoverJobs.setOnClickListener(v -> recoverServerJobs());
        job.addView(recoverJobs);
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

    private void pickAudioFile() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("audio/*");
        startActivityForResult(intent, REQ_PICK_AUDIO);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQ_PICK_AUDIO && resultCode == RESULT_OK && data != null && data.getData() != null) {
            importPickedAudio(data.getData());
        }
    }

    private void importPickedAudio(Uri uri) {
        io.execute(() -> {
            try {
                String displayName = displayNameForUri(uri);
                String safeName = safeFilename(displayName == null ? "picked-audio" : displayName);
                String mime = getContentResolver().getType(uri);
                File target = new File(getCacheDir(), System.currentTimeMillis() + "-" + safeName);
                try (InputStream in = getContentResolver().openInputStream(uri);
                     OutputStream out = new BufferedOutputStream(new java.io.FileOutputStream(target))) {
                    if (in == null) {
                        throw new IOException("Could not open selected audio.");
                    }
                    byte[] buf = new byte[8192];
                    int n;
                    while ((n = in.read(buf)) != -1) {
                        out.write(buf, 0, n);
                    }
                }
                runMain(() -> {
                    selectedAudioFile = target;
                    recordingFile = target;
                    selectedAudioMime = mime == null ? "application/octet-stream" : mime;
                    recordingText.setText("Selected audio: " + displayName);
                    uploadButton.setEnabled(true);
                });
            } catch (Exception e) {
                runMain(() -> recordingText.setText("Could not import audio: " + e.getMessage()));
            }
        });
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
            selectedAudioFile = null;
            selectedAudioMime = "audio/mp4";
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
            selectedAudioFile = recordingFile;
            selectedAudioMime = "audio/mp4";
            recordingText.setText("Saved recording: " + recordingFile.getName());
            uploadButton.setEnabled(recordingFile != null && recordingFile.isFile());
        } catch (RuntimeException e) {
            recordingText.setText("Recording was too short or failed.");
            if (recordingFile != null) {
                recordingFile.delete();
            }
            recordingFile = null;
            selectedAudioFile = null;
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
        File audio = selectedAudioFile != null ? selectedAudioFile : recordingFile;
        if (audio == null || !audio.isFile()) {
            recordingText.setText("Record or select audio first.");
            return;
        }
        persistSettings();
        uploadButton.setEnabled(false);
        setJob("Uploading...", 0);
        io.execute(() -> {
            try {
                JSONObject created = postMultipartJob(audio);
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

    private void recoverServerJobs() {
        persistSettings();
        setJob("Checking server jobs...", 0);
        io.execute(() -> {
            try {
                JSONObject jobs = requestJson("GET", "/api/jobs", null);
                JSONArray items = jobs.optJSONArray("jobs");
                JSONObject candidate = firstRecoverableJob(items);
                if (candidate == null) {
                    runMain(() -> setJob("No recoverable processing jobs found.", 0));
                } else {
                    String status = candidate.optString("status");
                    String jobId = candidate.optString("id");
                    String lessonId = candidate.optString("lesson_id", null);
                    if ("queued".equals(status) || "running".equals(status)) {
                        runMain(() -> setJob("Resuming job poll: " + jobId, (int) Math.round(candidate.optDouble("progress", 0))));
                        pollJob(jobId);
                        return;
                    }
                    if ("succeeded".equals(status) && lessonId != null && !lessonId.isEmpty()) {
                        lastLessonId = lessonId;
                        runMain(() -> {
                            setJob("Recovered completed job.", 100);
                            loadLessons();
                            openLesson(lessonId);
                        });
                    } else if ("failed".equals(status)) {
                        String error = candidate.optString("error", "Latest job failed");
                        runMain(() -> setJob(error, (int) Math.round(candidate.optDouble("progress", 0))));
                    }
                }
                JSONObject analysisJobs = requestJson("GET", "/api/analysis-jobs", null);
                JSONArray analysisItems = analysisJobs.optJSONArray("jobs");
                JSONObject analysis = firstRecoverableJob(analysisItems);
                if (analysis != null && "succeeded".equals(analysis.optString("status"))) {
                    String lessonId = analysis.optString("lesson_id", null);
                    if (lessonId != null && !lessonId.isEmpty()) {
                        runMain(() -> setServerStatus("Recovered latest analysis for " + lessonId));
                    }
                }
            } catch (Exception e) {
                runMain(() -> setJob("Job recovery failed: " + e.getMessage(), 0));
            }
        });
    }

    private JSONObject firstRecoverableJob(JSONArray jobs) {
        if (jobs == null) {
            return null;
        }
        JSONObject latestTerminal = null;
        for (int i = 0; i < jobs.length(); i++) {
            JSONObject job = jobs.optJSONObject(i);
            if (job == null) {
                continue;
            }
            String status = job.optString("status");
            if ("queued".equals(status) || "running".equals(status)) {
                return job;
            }
            if (latestTerminal == null) {
                latestTerminal = job;
            }
        }
        return latestTerminal;
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
        openedLessonId = id;
        detailPanel.addView(sectionTitle(meta.optString("profile", "Lesson")));
        detailPanel.addView(label(id, 13, MUTED, false));
        detailPanel.addView(label(lessonMetaLine(meta), 13, MUTED, false));

        HorizontalScrollView actionsScroll = new HorizontalScrollView(this);
        LinearLayout actions = new LinearLayout(this);
        actions.setOrientation(LinearLayout.HORIZONTAL);
        Button transcript = secondaryButton("Transcript");
        Button speakers = secondaryButton("Speakers");
        Button analysis = secondaryButton("Analysis");
        Button context = secondaryButton("Context");
        Button artifacts = secondaryButton("Artifacts");
        actions.addView(transcript);
        actions.addView(space(dp(8), 1));
        actions.addView(speakers);
        actions.addView(space(dp(8), 1));
        actions.addView(analysis);
        actions.addView(space(dp(8), 1));
        actions.addView(context);
        actions.addView(space(dp(8), 1));
        actions.addView(artifacts);
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
        context.setOnClickListener(v -> renderContext(content, lesson));
        artifacts.setOnClickListener(v -> renderArtifacts(content, lesson));
        renderTranscript(content, finalSegments);
    }

    private void renderTranscript(LinearLayout content, JSONArray segments) {
        content.removeAllViews();
        LinearLayout exports = new LinearLayout(this);
        exports.setOrientation(LinearLayout.HORIZONTAL);
        Button shareCleaned = secondaryButton("Share Cleaned");
        Button shareRaw = secondaryButton("Share Raw");
        Button viewRaw = secondaryButton("View Raw");
        Button viewHighlighted = secondaryButton("View Highlights");
        shareCleaned.setOnClickListener(v -> shareLessonExport(currentLessonIdFromPanel(), "/exports/transcript?variant=cleaned&format=txt", "text/plain"));
        shareRaw.setOnClickListener(v -> shareLessonExport(currentLessonIdFromPanel(), "/exports/transcript?variant=raw&format=txt", "text/plain"));
        exports.addView(shareCleaned, weightParams(1));
        exports.addView(space(dp(8), 1));
        exports.addView(shareRaw, weightParams(1));
        content.addView(exports);
        LinearLayout views = new LinearLayout(this);
        views.setOrientation(LinearLayout.HORIZONTAL);
        viewRaw.setOnClickListener(v -> loadTranscriptVariant(content, "raw"));
        viewHighlighted.setOnClickListener(v -> loadTranscriptVariant(content, "highlighted"));
        views.addView(viewRaw, weightParams(1));
        views.addView(space(dp(8), 1));
        views.addView(viewHighlighted, weightParams(1));
        content.addView(views);
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

    private void loadTranscriptVariant(LinearLayout content, String variant) {
        String lessonId = currentLessonIdFromPanel();
        content.removeAllViews();
        content.addView(label("Loading " + variant + " transcript...", 14, MUTED, false));
        io.execute(() -> {
            try {
                String text = requestText(
                        "GET",
                        "/api/lessons/" + encodePath(lessonId) + "/exports/transcript?variant=" + variant + "&format=txt",
                        null
                );
                runMain(() -> {
                    content.removeAllViews();
                    content.addView(sectionTitle(displayValue(variant) + " Transcript"));
                    Button share = secondaryButton("Share This Transcript");
                    share.setOnClickListener(v -> shareText(text, "text/plain", "Diarize transcript"));
                    content.addView(share);
                    content.addView(label(text.isEmpty() ? "(empty transcript)" : text, 14, TEXT, false));
                });
            } catch (Exception e) {
                runMain(() -> {
                    content.removeAllViews();
                    content.addView(label("Could not load transcript: " + e.getMessage(), 14, DANGER, false));
                });
            }
        });
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
                TextView status = label("Queueing analysis...", 14, MUTED, false);
                content.addView(status);
                analyze.setEnabled(false);
                io.execute(() -> {
                    try {
                        JSONObject created = requestJson(
                                "POST",
                                "/api/lessons/" + encodePath(lesson.optString("id")) + "/analysis-jobs",
                                payload.toString()
                        );
                        pollAnalysisJob(created.getString("id"), lesson.optString("id"), status, analyze);
                    } catch (Exception e) {
                        runMain(() -> {
                            analyze.setEnabled(true);
                            status.setText("Analysis failed: " + e.getMessage());
                            status.setTextColor(DANGER);
                        });
                    }
                });
            } catch (Exception e) {
                content.addView(label("Analysis failed: " + e.getMessage(), 14, DANGER, false));
            }
        });
        content.addView(analyze);

        Button shareAnalysis = secondaryButton("Share Analysis");
        shareAnalysis.setOnClickListener(v -> shareLessonExport(lesson.optString("id"), "/exports/analysis?format=txt", "text/plain"));
        content.addView(shareAnalysis);
    }

    private void renderContext(LinearLayout content, JSONObject lesson) {
        content.removeAllViews();
        content.addView(label("Loading context metrics...", 14, MUTED, false));
        String lessonId = lesson.optString("id");
        io.execute(() -> {
            try {
                JSONObject response = requestJson("GET", "/api/lessons/" + encodePath(lessonId) + "/context", null);
                runMain(() -> renderContextForm(content, lessonId, response));
            } catch (Exception e) {
                runMain(() -> {
                    content.removeAllViews();
                    content.addView(label("Could not load context: " + e.getMessage(), 14, DANGER, false));
                });
            }
        });
    }

    private void renderContextForm(LinearLayout content, String lessonId, JSONObject response) {
        content.removeAllViews();
        JSONObject context = response.optJSONObject("context_metrics");
        if (context == null) {
            context = new JSONObject();
        }
        String interpretation = response.optString("interpretation", "");
        content.addView(label(interpretation.isEmpty() ? "Context metrics" : interpretation, 14, TEXT, false));

        Map<String, EditText> fields = new LinkedHashMap<>();
        addMetricField(content, fields, context, "practice_hours_last_7_days", "Practice hours last 7 days");
        addMetricField(content, fields, context, "topic_difficulty", "Topic difficulty");
        addMetricField(content, fields, context, "idea_density", "Idea density");
        addMetricField(content, fields, context, "abstraction_level", "Abstraction");
        addMetricField(content, fields, context, "cognitive_branching", "Cognitive branching");
        addMetricField(content, fields, context, "technical_density", "Technical density");
        addMetricField(content, fields, context, "discourse_depth", "Discourse depth");
        addMetricField(content, fields, context, "lexical_retrieval_pressure", "Lexical pressure");
        addMetricField(content, fields, context, "fatigue_or_stress", "Fatigue or stress");
        addMetricField(content, fields, context, "long_pauses_per_min", "Long pauses/min");
        addMetricField(content, fields, context, "self_repairs_per_min", "Repairs/min");
        addMetricField(content, fields, context, "filled_pauses_per_min", "Filled pauses/min");

        Button save = primaryButton("Save Context");
        save.setOnClickListener(v -> saveContext(content, lessonId, fields));
        content.addView(save);

        TextView raw = label(prettyJson(context), 13, MUTED, false);
        content.addView(raw);
    }

    private void addMetricField(LinearLayout content, Map<String, EditText> fields, JSONObject context, String key, String title) {
        content.addView(fieldLabel(title));
        EditText edit = edit(context.isNull(key) ? "" : context.optString(key, ""), title);
        edit.setInputType(InputType.TYPE_CLASS_NUMBER | InputType.TYPE_NUMBER_FLAG_DECIMAL | InputType.TYPE_NUMBER_FLAG_SIGNED);
        fields.put(key, edit);
        content.addView(edit);
    }

    private void saveContext(LinearLayout content, String lessonId, Map<String, EditText> fields) {
        try {
            JSONObject payload = new JSONObject();
            for (String key : fields.keySet()) {
                String text = fields.get(key).getText().toString().trim();
                if (!text.isEmpty()) {
                    payload.put(key, Double.parseDouble(text));
                }
            }
            TextView status = label("Saving context...", 14, MUTED, false);
            content.addView(status);
            io.execute(() -> {
                try {
                    JSONObject updated = requestJson("PATCH", "/api/lessons/" + encodePath(lessonId) + "/context", payload.toString());
                    runMain(() -> renderContextForm(content, lessonId, updated));
                } catch (Exception e) {
                    runMain(() -> {
                        status.setText("Context save failed: " + e.getMessage());
                        status.setTextColor(DANGER);
                    });
                }
            });
        } catch (Exception e) {
            content.addView(label("Context save failed: " + e.getMessage(), 14, DANGER, false));
        }
    }

    private void renderArtifacts(LinearLayout content, JSONObject lesson) {
        content.removeAllViews();
        String lessonId = lesson.optString("id");
        Button shareCleaned = primaryButton("Share Cleaned Transcript");
        shareCleaned.setOnClickListener(v -> shareLessonExport(lessonId, "/exports/transcript?variant=cleaned&format=txt", "text/plain"));
        content.addView(shareCleaned);

        Button shareHighlighted = secondaryButton("Share Highlighted Transcript");
        shareHighlighted.setOnClickListener(v -> shareLessonExport(lessonId, "/exports/transcript?variant=highlighted&format=txt", "text/plain"));
        content.addView(shareHighlighted);

        Button shareSegments = secondaryButton("Share Segments JSON");
        shareSegments.setOnClickListener(v -> shareLessonExport(lessonId, "/exports/transcript?variant=cleaned&format=json", "application/json"));
        content.addView(shareSegments);

        Button shareAnalysis = secondaryButton("Share Analysis JSON");
        shareAnalysis.setOnClickListener(v -> shareLessonExport(lessonId, "/exports/analysis?format=json", "application/json"));
        content.addView(shareAnalysis);

        JSONArray artifacts = lesson.optJSONArray("artifacts");
        content.addView(label("Server artifacts", 16, TEXT, true));
        if (artifacts == null || artifacts.length() == 0) {
            content.addView(label("No artifact listing available.", 14, MUTED, false));
            return;
        }
        for (int i = 0; i < artifacts.length(); i++) {
            JSONObject artifact = artifacts.optJSONObject(i);
            if (artifact != null) {
                content.addView(label(
                        artifact.optString("name") + " - " + artifact.optString("filename") + " - " + formatBytes(artifact.optLong("size_bytes", 0)),
                        13,
                        MUTED,
                        false
                ));
            }
        }
    }

    private void pollAnalysisJob(String jobId, String lessonId, TextView statusView, Button analyzeButton) throws Exception {
        int consecutiveFailures = 0;
        while (!Thread.currentThread().isInterrupted()) {
            try {
                JSONObject job = requestJson("GET", "/api/analysis-jobs/" + encodePath(jobId), null);
                consecutiveFailures = 0;
                String status = job.optString("status", "unknown");
                String message = job.optString("message", status);
                int progress = (int) Math.round(job.optDouble("progress", 0));
                runMain(() -> statusView.setText(message + " (" + status + ", " + progress + "%)"));

                if ("succeeded".equals(status)) {
                    JSONObject updated = requestJson("GET", "/api/lessons/" + encodePath(lessonId), null);
                    runMain(() -> renderLesson(updated));
                    return;
                }
                if ("failed".equals(status)) {
                    String error = job.optString("error", "Analysis failed");
                    runMain(() -> {
                        analyzeButton.setEnabled(true);
                        statusView.setText(error);
                        statusView.setTextColor(DANGER);
                    });
                    return;
                }
            } catch (Exception e) {
                consecutiveFailures += 1;
                int failedPolls = consecutiveFailures;
                runMain(() -> statusView.setText("Connection interrupted; retrying (" + failedPolls + "/12): " + e.getMessage()));
                if (consecutiveFailures >= 12) {
                    runMain(() -> {
                        analyzeButton.setEnabled(true);
                        statusView.setText("Could not reach server after repeated retries: " + e.getMessage());
                        statusView.setTextColor(DANGER);
                    });
                    return;
                }
            }
            Thread.sleep(1600);
        }
    }

    private JSONObject postMultipartJob(File audio) throws Exception {
        String boundary = "DiarizeBoundary" + System.currentTimeMillis();
        String profile = profileInput.getText().toString().trim();
        if (profile.isEmpty()) {
            profile = "default";
        }
        URL url = new URL(serverUrl() + "/api/profiles/" + encodePath(profile) + "/jobs");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setDoOutput(true);
        conn.setConnectTimeout(15000);
        conn.setReadTimeout(60000);
        conn.setRequestProperty("Content-Type", "multipart/form-data; boundary=" + boundary);

        try (OutputStream out = new BufferedOutputStream(conn.getOutputStream())) {
            writeFormField(out, boundary, "model_size", selectedValue(modelInput));
            String language = selectedValue(languageInput);
            if (!language.isEmpty() && !"auto".equals(language)) {
                writeFormField(out, boundary, "language", language);
            }
            String speakerCount = selectedValue(speakerCountInput);
            if (!speakerCount.isEmpty()) {
                writeFormField(out, boundary, "num_speakers", speakerCount);
            }
            writeFormField(out, boundary, "backend", selectedValue(backendInput));
            writeFormField(out, boundary, "diarization_backend", selectedValue(diarizationInput));
            String batchSize = selectedValue(batchSizeInput);
            if (!batchSize.isEmpty()) {
                writeFormField(out, boundary, "batch_size", batchSize);
            }
            writeFileField(out, boundary, "audio", audio, selectedAudioMime == null ? "application/octet-stream" : selectedAudioMime);
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

    private String requestText(String method, String path, String body) throws Exception {
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
        return text;
    }

    private void shareLessonExport(String lessonId, String exportPath, String mimeType) {
        if (lessonId == null || lessonId.isEmpty()) {
            setServerStatus("Open a lesson first.");
            return;
        }
        setServerStatus("Preparing export...");
        io.execute(() -> {
            try {
                String text = requestText("GET", "/api/lessons/" + encodePath(lessonId) + exportPath, null);
                runMain(() -> shareText(text, mimeType, "Diarize export"));
            } catch (Exception e) {
                runMain(() -> setServerStatus("Export failed: " + e.getMessage()));
            }
        });
    }

    private void shareText(String text, String mimeType, String title) {
        Intent send = new Intent(Intent.ACTION_SEND);
        send.setType(mimeType == null ? "text/plain" : mimeType);
        send.putExtra(Intent.EXTRA_TEXT, text);
        startActivity(Intent.createChooser(send, title));
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

    private String currentLessonIdFromPanel() {
        return openedLessonId == null ? "" : openedLessonId;
    }

    private String encodePath(String value) throws IOException {
        return URLEncoder.encode(value == null ? "" : value, "UTF-8").replace("+", "%20");
    }

    private String displayNameForUri(Uri uri) {
        String name = null;
        try (Cursor cursor = getContentResolver().query(uri, null, null, null, null)) {
            if (cursor != null && cursor.moveToFirst()) {
                int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                if (index >= 0) {
                    name = cursor.getString(index);
                }
            }
        } catch (Exception ignored) {
        }
        if (name == null || name.trim().isEmpty()) {
            String path = uri.getLastPathSegment();
            name = path == null || path.trim().isEmpty() ? "audio" : path;
        }
        return name;
    }

    private String safeFilename(String value) {
        String safe = String.valueOf(value == null ? "audio" : value)
                .trim()
                .replaceAll("[^A-Za-z0-9._-]+", "_")
                .replaceAll("^_+|_+$", "");
        return safe.isEmpty() ? "audio" : safe;
    }

    private void persistSettings() {
        prefs.edit()
                .putString("server_url", serverUrl())
                .putString("profile", profileInput.getText().toString().trim())
                .putString("model", selectedValue(modelInput))
                .putString("language", selectedValue(languageInput))
                .putString("speaker_count", selectedValue(speakerCountInput))
                .putString("backend", selectedValue(backendInput))
                .putString("diarization", selectedValue(diarizationInput))
                .putString("batch_size", selectedValue(batchSizeInput))
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
        if ("raw".equals(value)) {
            return "Raw";
        }
        if ("highlighted".equals(value)) {
            return "Highlighted";
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
        if ("auto".equals(value)) {
            return "Auto";
        }
        if ("mlx".equals(value)) {
            return "MLX Whisper";
        }
        if ("whisper_mps".equals(value)) {
            return "Whisper MPS";
        }
        if ("whisperx".equals(value)) {
            return "WhisperX";
        }
        if ("pyannote".equals(value)) {
            return "Speaker diarization";
        }
        if ("single_speaker".equals(value)) {
            return "Single speaker";
        }
        return value;
    }

    private String lessonMetaLine(JSONObject meta) {
        StringBuilder line = new StringBuilder();
        if (meta == null) {
            return "";
        }
        if (meta.has("duration_sec")) {
            line.append("Duration ").append(formatDuration(meta.optDouble("duration_sec", 0)));
        }
        if (meta.has("num_segments")) {
            if (line.length() > 0) line.append(" - ");
            line.append(meta.optInt("num_segments", 0)).append(" segments");
        }
        if (meta.has("num_speakers")) {
            if (line.length() > 0) line.append(" - ");
            line.append(meta.optInt("num_speakers", 0)).append(" speakers");
        }
        String asr = meta.optString("asr_backend", "");
        String diar = meta.optString("diarization_backend", "");
        if (!asr.isEmpty() || !diar.isEmpty()) {
            if (line.length() > 0) line.append("\n");
            line.append("ASR ").append(asr.isEmpty() ? "?" : displayValue(asr));
            line.append(" - Diarization ").append(diar.isEmpty() ? "?" : displayValue(diar));
        }
        String provider = meta.optString("llm_provider", "");
        String model = meta.optString("llm_model", "");
        if (!provider.isEmpty() || !model.isEmpty()) {
            if (line.length() > 0) line.append("\n");
            line.append("LLM ").append(provider.isEmpty() ? "?" : provider);
            if (!model.isEmpty()) {
                line.append(" / ").append(model);
            }
        }
        return line.toString();
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

    private String formatBytes(long bytes) {
        if (bytes <= 0) {
            return "0 B";
        }
        String[] units = new String[]{"B", "KB", "MB", "GB"};
        int index = 0;
        double value = bytes;
        while (value >= 1024 && index < units.length - 1) {
            value /= 1024.0;
            index += 1;
        }
        return String.format(Locale.US, index == 0 ? "%.0f %s" : "%.1f %s", value, units[index]);
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
