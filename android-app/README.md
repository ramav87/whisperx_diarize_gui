# Diarize Mobile Android

Native Android client for the `diarize-server` API.

## Current features

- Connect to a LAN `diarize-server` URL.
- Request microphone permission.
- Record audio as M4A with Android `MediaRecorder`.
- Pick an existing audio file from Android's document picker.
- Select server processing settings, including ASR backend, diarization backend,
  model size, language, speaker count, and batch size.
- Upload the recording to `POST /api/profiles/{profile_id}/jobs`.
- Poll job progress with `GET /api/jobs/{job_id}`.
- Recover persisted server job state after reconnecting.
- Refresh and open lessons from `GET /api/lessons`.
- Show transcript segments from `GET /api/lessons/{lesson_id}` and view
  cleaned/raw/highlighted transcript exports.
- Save speaker labels and student speaker selection.
- Trigger durable lesson AI analysis jobs and poll progress.
- View and edit server-backed context metrics.
- Share cleaned/raw/highlighted transcripts, segment JSON, and analysis exports.

## Run

1. Start the Python API server:

   ```bash
   diarize-server
   ```

2. Open this `android-app/` directory in Android Studio.
3. Let Gradle sync and install any Android SDK components it requests.
4. Run the app on the tablet.
5. Set the server URL to the server machine's LAN URL, for example:

   ```text
   http://192.168.1.160:8000
   ```

The app permits cleartext HTTP for local network development. Do not expose that
configuration outside a trusted LAN.
