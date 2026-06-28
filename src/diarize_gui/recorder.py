import os
from datetime import datetime
from typing import Optional, Callable
import soundfile as sf
from tkinter import messagebox
import numpy as np
import threading

try:
    import sounddevice as sd
    SOUNDDEVICE_IMPORT_ERROR = None
except (ImportError, OSError) as exc:
    sd = None
    SOUNDDEVICE_IMPORT_ERROR = exc


class AudioRecorder:
    """
    Simple audio recorder using sounddevice + soundfile.
    Records mono audio into a WAV file. The processing pipeline normalizes the
    result to 16 kHz before transcription.
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
        self.recorded_at_time = None

        self.is_recording: bool = False
        self.record_file: Optional[sf.SoundFile] = None
        self.record_stream: Optional[sd.InputStream] = None
        self.recorded_file_path: Optional[str] = None
        self._level_lock = threading.Lock()
        self._rms = 0.0
        self._peak = 0.0


    def _set_status(self, text: str):
        if self.on_status is not None:
            self.on_status(text)

    def _require_sounddevice(self) -> bool:
        if sd is not None:
            return True

        detail = str(SOUNDDEVICE_IMPORT_ERROR) if SOUNDDEVICE_IMPORT_ERROR else "sounddevice is unavailable"
        msg = (
            "Live recording is unavailable because PortAudio is not installed.\n\n"
            f"{detail}\n\n"
            "On Ubuntu, install it with:\n"
            "sudo apt-get install libportaudio2"
        )
        self._set_status("Recording unavailable: PortAudio is not installed")
        messagebox.showerror("Recording unavailable", msg)
        return False

    def list_input_devices(self):
        devices = []
        if sd is None:
            self._set_status("Recording unavailable: PortAudio is not installed")
            return devices

        try:
            all_devices = sd.query_devices()
            for idx, dev in enumerate(all_devices):
                if dev.get("max_input_channels", 0) > 0:
                    devices.append({"index": idx, "name": dev["name"]})
        except Exception as e:
            messagebox.showerror("Error querying audio devices", str(e))
        return devices

    def _candidate_samplerates(self, device_index: Optional[int]) -> list[int]:
        rates: list[int] = []
        if sd is not None:
            try:
                dev = sd.query_devices(device_index, "input") if device_index is not None else sd.query_devices(kind="input")
                default_rate = int(float(dev.get("default_samplerate", 0) or 0))
                if default_rate > 0:
                    rates.append(default_rate)
            except Exception:
                pass

        rates.extend([self.samplerate, 48000, 44100, 32000, 22050])
        deduped: list[int] = []
        for rate in rates:
            if rate > 0 and rate not in deduped:
                deduped.append(rate)
        return deduped

    def _select_samplerate(self, device_index: Optional[int]) -> int:
        errors: list[str] = []
        for samplerate in self._candidate_samplerates(device_index):
            try:
                sd.check_input_settings(
                    device=device_index,
                    channels=self.channels,
                    samplerate=samplerate,
                )
                return samplerate
            except Exception as exc:
                errors.append(f"{samplerate} Hz: {exc}")

        detail = "\n".join(errors) if errors else "No sample rates were accepted."
        raise RuntimeError(f"Could not open input device with a supported sample rate.\n{detail}")
    
    def get_level(self):
        """Returns (rms, peak) in 0..1 range."""
        with self._level_lock:
            return self._rms, self._peak


    def start_recording(self, output_dir: str, device_index: Optional[int] = None):
        if self.is_recording:
            return

        if not self._require_sounddevice():
            return

        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_started_at = datetime.now().isoformat(timespec="seconds")
        filename = f"recording_{timestamp}.wav"
        self.recorded_file_path = os.path.join(output_dir, filename)
        self.recorded_at_time = recording_started_at

        try:
            def callback(indata, frames, time, status):
                if status:
                    print(status)
                self.record_file.write(indata)
                
                # Compute RMS/peak for meter (mono or multi-channel)
                try:
                    x = indata
                    if hasattr(x, "shape") and len(x.shape) == 2:
                        x = x[:, 0]  # first channel
                    x = np.asarray(x, dtype=np.float32)

                    peak = float(np.max(np.abs(x))) if x.size else 0.0
                    rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0

                    with self._level_lock:
                        # smooth a bit so it doesn't jitter
                        self._peak = max(peak, self._peak * 0.85)
                        self._rms = (rms * 0.25) + (self._rms * 0.75)
                except Exception:
                    pass

            recording_samplerate = self._select_samplerate(device_index)
            self.record_file = sf.SoundFile(
                self.recorded_file_path,
                mode="w",
                samplerate=recording_samplerate,
                channels=self.channels,
                subtype="PCM_16",
            )
            self.record_stream = sd.InputStream(
                samplerate=recording_samplerate,
                channels=self.channels,
                callback=callback,
                device=device_index,
            )

            self.record_stream.start()
            self.is_recording = True
            self._set_status(f"Recording... ({recording_samplerate} Hz)")

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
