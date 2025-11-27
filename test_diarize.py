import whisperx, pandas as pd
from pyannote.audio import Pipeline
import os

audio_path = "/Users/rama/Desktop/Oct15_lesson_short.mp3"
token = os.environ["HUGGINGFACE_TOKEN"]

# small repro of just diarization + assignment
model = whisperx.load_model("small", device="cpu", compute_type="int8")
audio = whisperx.load_audio(audio_path)
res = model.transcribe(audio)

align_model, meta = whisperx.load_align_model(language_code=res["language"], device="cpu")
aligned = whisperx.align(res["segments"], align_model, meta, audio, "cpu")

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.0", use_auth_token=token)
ann = pipe(audio_path)

segs = []
for seg, _, spk in ann.itertracks(yield_label=True):
    segs.append({"start": seg.start, "end": seg.end, "speaker": spk})
df = pd.DataFrame(segs)

out = whisperx.assign_word_speakers(df, aligned)
print(out["segments"][0:3])
