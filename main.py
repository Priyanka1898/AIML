import torchaudio
import os
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login
from speechbrain.inference.TTS import FastSpeech2
from speechbrain.inference.vocoders import HIFIGAN
import logging

# Suppress SpeechBrain warnings
logging.getLogger("speechbrain").setLevel(logging.ERROR)
from huggingface_hub import login

# Authenticate with Hugging Face before loading models
import os
from huggingface_hub import login

token = "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"  # Replace with your actual token
os.environ["HUGGINGFACE_TOKEN"] = token
login(token=token)


# Initialize SpeechBrain FastSpeech2 and HiFi-GAN
fastspeech2 = FastSpeech2.from_hparams(
    source="speechbrain/tts-fastspeech2-ljspeech",
    savedir="pretrained_models/tts-fastspeech2-ljspeech"
)

# Suppress encoder length warning
fastspeech2.hparams.text_encoder.ignore_len()


hifi_gan = HIFIGAN.from_hparams(
    source="speechbrain/tts-hifigan-ljspeech", 
    savedir="pretrained_models/tts-hifigan-ljspeech"
)

# Run TTS with text input
input_text = "This is a test sentence for FastSpeech2."

mel_output, durations, pitch, energy = fastspeech2.encode_text(
  [input_text], pace=1.0, pitch_rate=1.0, energy_rate=1.0
)

print(dir(fastspeech2.hparams))
# Convert spectrogram to waveform
waveforms = hifi_gan.decode_batch(mel_output)

# Save waveform
torchaudio.save('output_text.wav', waveforms.squeeze(1), 22050)

print("✅ TTS audio saved: output_text.wav")
