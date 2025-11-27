import os
from datetime import datetime
from typing import Optional, Callable

import sounddevice as sd
import soundfile as sf
from tkinter import messagebox


class AudioRecorder:
    """
    Simple audio recorder using sounddevice + soundfile.
    Records mono 16 kHz audio into a WAV file.
    """

    def __init__(
        self,
        samplerate: int = 16000,
        channels: int = 1,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.samplerate = samplerate
        self.channels = channels
        self.on_status = on_status

        self.is_recording: bool = False
        self.record_file: Optional[sf.SoundFile] = None
        self.record_stream: Optional[sd.InputStream] = None
        self.recorded_file_path: Optional[str] = None

    def _set_status(self, text: str):
        if self.on_status is not None:
            self.on_status(text)

    def list_input_devices(self):
        devices = []
        try:
            all_devices = sd.query_devices()
            for idx, dev in enumerate(all_devices):
                if dev.get("max_input_channels", 0) > 0:
                    devices.append({"index": idx, "name": dev["name"]})
        except Exception as e:
            messagebox.showerror("Error querying audio devices", str(e))
        return devices

    def start_recording(self, output_dir: str, device_index: Optional[int] = None):
        if self.is_recording:
            return

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recording_{timestamp}.wav"
        self.recorded_file_path = os.path.join(output_dir, filename)

        try:
            self.record_file = sf.SoundFile(
                self.recorded_file_path,
                mode="w",
                samplerate=self.samplerate,
                channels=self.channels,
                subtype="PCM_16",
            )

            def callback(indata, frames, time, status):
                if status:
                    print(status)
                self.record_file.write(indata)

            self.record_stream = sd.InputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                callback=callback,
                device=device_index,
            )

            self.record_stream.start()
            self.is_recording = True
            self._set_status("Recording...")

        except Exception as e:
            self._set_status("Error starting recording")
            messagebox.showerror("Error starting recording", str(e))
            self.record_file = None
            self.record_stream = None
            self.is_recording = False
            self.recorded_file_path = None

    def stop_recording(self):
        if not self.is_recording:
            return

        try:
            if self.record_stream is not None:
                self.record_stream.stop()
                self.record_stream.close()
            if self.record_file is not None:
                self.record_file.close()
        except Exception as e:
            messagebox.showerror("Error stopping recording", str(e))

        self.record_stream = None
        self.record_file = None
        self.is_recording = False
        self._set_status("Recording stopped")

        return self.recorded_file_path
