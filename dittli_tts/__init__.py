import os

import soundfile as sf
import torch

from dittli_tts.infer import load_engine
from dittli_tts.models.synthesizer import VoiceSynthesizer as VoiceSynthesizer
from dittli_tts.nn import commons
from dittli_tts.text import phonemes_to_ids
from dittli_tts.text.english import grapheme_to_phoneme, normalize_text
from dittli_tts.text.symbols import symbols as symbols
from dittli_tts.utils.config import (
    ADD_BLANK,
    SAMPLING_RATE,
    SPK2ID,
)


class DittliTTS:
    def __init__(self, checkpoint_path, device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        self.model = load_engine(checkpoint_path, self.device)

    def speak(self, text, output_path="output.wav", speaker="MALE", speed=1.0):
        """Synthesize text to speech and save to output_path."""
        print(f"Synthesizing: {text}")

        # Normalize text
        normalized = normalize_text(text)

        # Phonemize
        phones, tones, word2ph = grapheme_to_phoneme(normalized)

        # Convert to sequence
        phone_ids, tone_ids, lang_ids = phonemes_to_ids(phones, tones, "EN")

        # Add blanks
        if ADD_BLANK:
            phone_ids = commons.insert_blanks(phone_ids, 0)
            tone_ids = commons.insert_blanks(tone_ids, 0)
            lang_ids = commons.insert_blanks(lang_ids, 0)

        x = torch.LongTensor(phone_ids).unsqueeze(0).to(self.device)
        x_lengths = torch.LongTensor([len(phone_ids)]).to(self.device)
        tone = torch.LongTensor(tone_ids).unsqueeze(0).to(self.device)
        language = torch.LongTensor(lang_ids).unsqueeze(0).to(self.device)

        # Speaker ID
        if speaker not in SPK2ID:
            print(f"Warning: Speaker '{speaker}' not found, using ID 0. Available: {list(SPK2ID.keys())}")
            sid = torch.LongTensor([0]).to(self.device)
        else:
            sid = torch.LongTensor([SPK2ID[speaker]]).to(self.device)

        # BERT features (disabled - using zero tensors)
        bert = torch.zeros(1024, len(phone_ids)).to(self.device).unsqueeze(0)
        ja_bert = torch.zeros(768, len(phone_ids)).to(self.device).unsqueeze(0)

        # speed > 1.0 = faster speech, < 1.0 = slower speech
        length_scale = 1.0 / speed

        with torch.no_grad():
            audio, *_ = self.model.infer(
                x, x_lengths, sid, tone, language, bert, ja_bert,
                noise_scale=0.667,
                noise_scale_w=0.8,
                length_scale=length_scale
            )

        audio_np = audio[0, 0].cpu().numpy()
        sf.write(output_path, audio_np, SAMPLING_RATE)
        print(f"Saved audio to {output_path}")
        return audio_np
