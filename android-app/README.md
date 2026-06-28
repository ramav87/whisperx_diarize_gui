# Diarize Mobile Android

Native Android client for the `diarize-server` API.

## Current features

- Connect to a LAN `diarize-server` URL.
- Request microphone permission.
- Record audio as M4A with Android `MediaRecorder`.
- Upload the recording to `POST /api/jobs`.
- Poll job progress with `GET /api/jobs/{job_id}`.
- Refresh and open lessons from `GET /api/lessons`.
- Show transcript segments from `GET /api/lessons/{lesson_id}`.
- Save speaker labels and student speaker selection.
- Trigger lesson AI analysis.

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
