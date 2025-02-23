# import torchaudio
# import os
# import os
# from transformers import AutoModel, AutoTokenizer

# # Set your Hugging Face token
# os.environ["HUGGINGFACE_TOKEN"] = "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"

# # Load the model with authentication
# model_name = "bert-base-uncased"
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=os.environ["HUGGINGFACE_TOKEN"])
# model = AutoModel.from_pretrained(model_name, use_auth_token=os.environ["HUGGINGFACE_TOKEN"])

# from speechbrain.inference.TTS import FastSpeech2
# from speechbrain.inference.vocoders import HIFIGAN

# # Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
# fastspeech2 = FastSpeech2.from_hparams(
#     source="speechbrain/tts-fastspeech2-ljspeech", 
#     savedir="pretrained_models/tts-fastspeech2-ljspeech",
#     use_auth_token=os.environ["HUGGINGFACE_TOKEN"]  # Pass the token
# )

# hifi_gan = HIFIGAN.from_hparams(
#     source="speechbrain/tts-hifigan-ljspeech", 
#     savedir="pretrained_models/tts-hifigan-ljspeech",
#     use_auth_token=os.environ["HUGGINGFACE_TOKEN"]
# )

# # fastspeech2 = FastSpeech2.from_hparams(source="speechbrain/tts-fastspeech2-ljspeech", savedir="pretrained_models/tts-fastspeech2-ljspeech")
# # hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="pretrained_models/tts-hifigan-ljspeech")

# # Run TTS with text input
# input_text = "were the leaders in this luckless change; though our own Baskerville; who was at work some years before them; went much on the same lines;"

# mel_output, durations, pitch, energy = fastspeech2.encode_text(
#   [input_text],
#   pace=1.0,        # scale up/down the speed
#   pitch_rate=1.0,  # scale up/down the pitch
#   energy_rate=1.0, # scale up/down the energy
# )

# # Running Vocoder (spectrogram-to-waveform)
# waveforms = hifi_gan.decode_batch(mel_output)

# # Save the waverform
# torchaudio.save('example_TTS_input_text.wav', waveforms.squeeze(1), 22050)


# # Run TTS with phoneme input
# input_phonemes = ['W', 'ER', 'DH', 'AH', 'L', 'IY', 'D', 'ER', 'Z', 'IH', 'N', 'DH', 'IH', 'S', 'L', 'AH', 'K', 'L', 'AH', 'S', 'CH', 'EY', 'N', 'JH', 'spn', 'DH', 'OW', 'AW', 'ER', 'OW', 'N', 'B', 'AE', 'S', 'K', 'ER', 'V', 'IH', 'L', 'spn', 'HH', 'UW', 'W', 'AA', 'Z', 'AE', 'T', 'W', 'ER', 'K', 'S', 'AH', 'M', 'Y', 'IH', 'R', 'Z', 'B', 'IH', 'F', 'AO', 'R', 'DH', 'EH', 'M', 'spn', 'W', 'EH', 'N', 'T', 'M', 'AH', 'CH', 'AA', 'N', 'DH', 'AH', 'S', 'EY', 'M', 'L', 'AY', 'N', 'Z', 'spn']
# mel_output, durations, pitch, energy = fastspeech2.encode_phoneme(
#   [input_phonemes],
#   pace=1.0,        # scale up/down the speed
#   pitch_rate=1.0,  # scale up/down the pitch
#   energy_rate=1.0, # scale up/down the energy
# )


# # Running Vocoder (spectrogram-to-waveform)
# waveforms = hifi_gan.decode_batch(mel_output)

# # Save the waverform
# torchaudio.save('example_TTS_input_phoneme.wav', waveforms.squeeze(1), 22050)

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
