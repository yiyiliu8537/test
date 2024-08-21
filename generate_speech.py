import os
import torch
import soundfile as sf
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

def generate_speech( infer_dataset , output_dir )
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    model = SpeechT5ForTextToSpeech.from_pretrained(
        "Olivia111/speecht5_finetuned_en32_lr401"
    )

    output_dir = '/content/drive/MyDrive/thesis_2/dataset/baseline_wav/'
    os.makedirs(output_dir, exist_ok=True)

    for i in range( len( infer_dataset)  ):
        inputs = processor(text=infer_dataset[i]['raw_text'], return_tensors="pt")
        speech = model.generate_speech(inputs["input_ids"], infer_dataset[i][speaker_embeddings], vocoder=vocoder)
        output_wav_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_baseline.wav")
        sf.write(output_wav_path, speech.cpu().numpy(), 16000)


__main__()
    generate_speech("microsoft/speecht5_tts", )
    generate_speech("microsoft/speecht5_tts", )
    generate_speech("microsoft/speecht5_tts", )
    generate_speech("microsoft/speecht5_tts", )

