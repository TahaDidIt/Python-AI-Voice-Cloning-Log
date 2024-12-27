#Set-up
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu = True)

#Generate speech via zero-shot cloning and default settings
"""
#Single reference
tts.tts_to_file(text = "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path = "test1output.wav",
                speaker_wav = "C:/Taha/projects/Taha Programming/Python AI Voice Cloning/XTTS v2/ranni_ref_10sec.wav",
                language = "en",
                split_sentences = True)
"""
#Multi reference
tts.tts_to_file(text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                file_path="output.wav",
                speaker_wav=["C:/Taha/projects/Taha Programming/Python AI Voice Cloning/XTTS v2/ranni_ref_8sec.wav",
                             "C:/Taha/projects/Taha Programming/Python AI Voice Cloning/XTTS v2/ranni_ref_10sec.wav",
                             "C:/Taha/projects/Taha Programming/Python AI Voice Cloning/XTTS v2/ranni_ref_12sec.wav"],
                language="en")
