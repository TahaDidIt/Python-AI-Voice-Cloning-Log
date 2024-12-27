import os
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts



# Add here the xtts_config path
CONFIG_PATH = "C:/Users/Taha/TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_LJSpeech_FT_Ranni1-November-16-2024_05+46PM-dbf1a08a/config.json"
# Add here the vocab file that you have used to train the model
TOKENIZER_PATH = "C:/Users/Taha/TTS/recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/vocab.json"
# Add here the checkpoint that you want to do inference with
XTTS_CHECKPOINT = "C:/Users/Taha/TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_LJSpeech_FT_Ranni1-November-16-2024_05+46PM-dbf1a08a/best_model.pth"
# Add here the speaker reference
SPEAKER_REFERENCE = "C:/Taha/projects/Taha Programming/Python AI Voice Cloning/datasets/Ranni 22050Hz LJS/wavs/audio153.wav"

# output wav path
OUTPUT_WAV_PATH = "xtts-ft-temp3.wav"

"""
NOTES: audio153 is a good ref to use. audio41 (inkin thee an invitation...) causes noticable noise.
However, RVC does smooth out this noise.
"""




print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])

print("Inference...")
out = model.inference(
    #"It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
    #"Ah, a dictionary in Python, much akin to a repository of paired knowledge, doth serve as a mutable collection of key-value pairs. Each key acts as a unique identifier, whilst its corresponding value holds the data associated with it. This structure alloweth for swift retrieval, insertion, or modification of data, provided the key is known. One might envision it as a tome of arcane lore, where each entry—the key—is inscribed with a glyph, and the corresponding description— the value— containeth the wisdom bound thereto.",
    #"Ah, the blue of the sky, a wonder wrought by the intricate dance of light. The azure hue cometh from the scattering of sunlight by particles within the atmosphere—a phenomenon known as Rayleigh scattering",
    #"Be kind to thyself. Sit with this feeling, but do not let it linger overlong. Even the faintest light of dawn scatters the gloomiest night. Seek what soothes thee—a tender thought, a melody, or the company of those thou trust. This too shall pass, as the river never ceases its flow.",
    "Think, dear Tarnished, before thy gold meets its doom!",
    "en",
    gpt_cond_latent,
    speaker_embedding,
    temperature=0.7, # Add custom parameters here
)
torchaudio.save(OUTPUT_WAV_PATH, torch.tensor(out["wav"]).unsqueeze(0), 24000)