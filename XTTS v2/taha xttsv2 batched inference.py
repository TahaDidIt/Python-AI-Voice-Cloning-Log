import os
import re   # regular expressions, for splitting text file into sentences
import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

"""
This is adapted code form the advanced inference test, made to work with a text file that holds multiple sentences,
running inference on each sentence separately.

STRUCTURE:
1) Config paths
2) Setup model
3) Input text/ text parsing and slicing
4) Inference
"""






### CONFIG PATHS


# Add here the xtts_config path
CONFIG_PATH = "C:/Users/Taha/TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_LJSpeech_FT_Ranni1-November-16-2024_05+46PM-dbf1a08a/config.json"
# Add here the vocab file that you have used to train the model
TOKENIZER_PATH = "C:/Users/Taha/TTS/recipes/ljspeech/xtts_v2/run/training/XTTS_v2.0_original_model_files/vocab.json"
# Add here the checkpoint that you want to do inference with
XTTS_CHECKPOINT = "C:/Users/Taha/TTS/recipes/ljspeech/xtts_v2/run/training/GPT_XTTS_v2.0_LJSpeech_FT_Ranni1-November-16-2024_05+46PM-dbf1a08a/best_model.pth"
# Add here the speaker reference
SPEAKER_REFERENCE = "C:/Taha/projects/Taha Programming/Python AI Voice Cloning/datasets/Ranni 22050Hz LJS/wavs/audio153.wav"
# Sample rate to use for audio output and waveform generation for silences in between batched audio
sample_rate = 24000

# output wav path - tweaked to put files into batch inference folder
# Old version: OUTPUT_WAV_PATH = "xtts-ft-temp3.wav"
base_dir = os.path.dirname(os.path.abspath(__file__))
# Output dir instead of singular wav path as later we save each wav to this dir
output_dir = os.path.join(base_dir, "batched inference output")

"""
NOTES: audio153 is a good ref to use. audio41 ("inkin thee an invitation...") causes noticable noise.
However, RVC does smooth out this noise.
"""







### MODEL SETUP

print("Loading model...")
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=XTTS_CHECKPOINT, vocab_path=TOKENIZER_PATH, use_deepspeed=False)
model.cuda()

print("Computing speaker latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[SPEAKER_REFERENCE])






### INPUT TEXT - Preset list version
"""
# preset version - Input text as list
input_text = [
    "Be kind to thyself.",
    "Sit with this feeling, but do not let it linger overlong.",
    "Even the faintest light of dawn scatters the gloomiest night.",
    "Seek what soothes thee—a tender thought, a melody, or the company of those thou trust.",
    "This too shall pass, as the river never ceases its flow.",
]
"""






### INPUT TEXT - text file version: Parse and split into sentence

# open txt file
with open(os.path.join(output_dir, "sentences.txt"), "r", encoding = "utf-8") as file:
    text = file.read()

# split the sentences apart
"""
HOW THIS WORKS:
1. We use the `re` module for splitting text based on punctuation.
2. The regex `(\.{3}|[.!?])` ensures we handle:
    - Ellipses (`...`) as a single punctuation unit.
    - or Full stops (`.`), exclamation marks (`!`), and question marks (`?`).
3. Capturing groups `()` in the regex allow punctuation to be retained in the results rather than lost upon splitting.
4. Whitespace and empty strings are removed by strip() to ensure clean results.
5. How the if condition for .strip() works:
    - Python accepts an empty sentence ('') as a result of .strip() as False, anything else true
    - So it gets a true for non empty sentences and knows it can strip
6. After splitting, we combine each sentence with its respective punctuation by iterating
   through the list in pairs (sentence + punctuation) for the sake of the TTS engine.
"""
input_text = [sentence.strip() for sentence in re.split(r"(\.{3}|[.!?])", text) if sentence.strip()]

# combine sentences back with their punctuation mark
sentences = [
    input_text[i] + input_text[i + 1]
    for i in range(0, len(input_text) - 1, 2)  # range(inclusive, exclusive, step)
    #We stop it 1 element before the end to prevent error if last element has no punc.
]
print(sentences)






### BATCHED INFERENCE - Single output file

# Generate silence between sentences
silence_duration = 0.3
silence_waveform = torch.zeros(int(sample_rate * silence_duration))

# Prepare final audio file
OUTPUT_WAV_PATH = os.path.join(output_dir, "joined_audio_output.wav")
# Initialise final audio to append to
joined_audio = []


#Inference
print("Performing BATCHED inference...")
for sentence in sentences:
    out = model.inference(
        sentence,
        "en",
        gpt_cond_latent,
        speaker_embedding,
        # Add custom parameters here
        temperature = 0.7,
        speed = 0.95
    )

    #hold the output to a variable so that we can store it and append final joined output file
    temp_waveform = torch.tensor(out["wav"])

    joined_audio.append(temp_waveform)
    joined_audio.append(silence_waveform)

print("Batched inference complete! Creating final audio file...")


# Concatinate all the audios into 1 audio
joined_audio = torch.cat(joined_audio)
# Save audio output file
torchaudio.save(OUTPUT_WAV_PATH, joined_audio.unsqueeze(0), sample_rate)

print("")
print(f"Saved audio to {OUTPUT_WAV_PATH}")






"""
### BATCHED INFERENCE - separate output files

print("Performing BATCHED inference...")
for i, text in enumerate(sentences):
    out = model.inference(
        text,
        "en",
        gpt_cond_latent,
        speaker_embedding,
        temperature = 0.7 # Add custom parameters here
    )
    # Saving each output file each time
    OUTPUT_WAV_PATH = os.path.join(output_dir, f"output_{i + 1}.wav")
    torchaudio.save(OUTPUT_WAV_PATH, torch.tensor(out["wav"]).unsqueeze(0), 24000)
    print(f"Saved audio to {OUTPUT_WAV_PATH}")

print("Batched inference complete!")
"""


""" OLD - Normal inference

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
"""
