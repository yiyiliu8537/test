import os
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifieir
import librosa
import json
from pathlib import Path
from tqdm import tqdm
import time
from datasets import Dataset, Audio, Features, Value
from datasets import load_dataset
import scipy.signal

#this is for trainning data test data from array 
def get_speaker_embedding(array_wave):
    model_name = "speechbrain/spkrec-xvect-voxceleb"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    speaker_embedding_model = EncoderClassifier.from_hparams(
        source=spk_model_name,
        run_opts={"device": device},
        savedir=os.path.join("/tmp", model_name),
    
    speaker_embeddings = speaker_embedding_model.encode_batch(torch.tensor(arry_waveform))
    speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
    speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

#this use for baseline from wave audio file  
def get_speaker_embedding_2( wav_file ):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="pretrained_models/spkrec-xvect-voxceleb")
    signal, fs = torchaudio.load( wav_file )
    embeddings = classifier.encode_batch( signal )
    return embedings

def get_array( ):
    y, src_sample_rate =high_quality_resample( wav_path , 42000 , 16000 )
    array = y.tolist()
    return array

def high_quality_resample(audio_path, orig_sr, target_sr):
    y, sr = librosa.load(audio_path, sr=orig_sr)
    y_resampled = scipy.signal.resample_poly(y, target_sr, orig_sr)
    return y_resampled, target_sr

def get_dataSet( wav_dir='/content/drive/MyDrive/thesis_2/dataset/train_dataset/RRBI/wav' , wav_dir='/content/drive/MyDrive/thesis_2/dataset/train_dataset/RRBI/transcript' ):

    wav_dir = '/content/drive/MyDrive/thesis_2/dataset/train_dataset/RRBI/wav'
    text_dir = '/content/drive/MyDrive/thesis_2/dataset/train_dataset/RRBI/transcript'

    wav_files = sorted([f for f in os.listdir(wav_dir) if f.endswith('.wav')])
    text_files = sorted([f for f in os.listdir(text_dir) if f.endswith('.txt')])

    assert len(wav_files) == len(text_files), "Mismatch in number of WAV and text files"

    english_language_id = 0

    data_list = []

    for wav_file, text_file in tqdm(zip(wav_files, text_files), total=len(wav_files)):
        #get audio_id
        file_stat = os.stat(os.path.join(wav_dir, wav_file))
        creation_time = time.strftime('%Y%m%d-%H%M', time.localtime(file_stat.st_ctime))
        modification_time = time.strftime('%Y%m%d-%H:%M:%S', time.localtime(file_stat.st_mtime))
        audio_id = f"{creation_time}-PLENARY-3-en_{modification_time}_4"
        #get raw_text  
        text_path = os.path.join(text_dir, text_file)

        with open(text_path, 'r') as file:
            raw_text = file.read().strip()
	#get array
        wav_path = os.path.join(wav_dir, wav_file)
        temp_array = get_array( wav_path ,  orig_sr , target_sr )
        # data structure 
        formatted_datawav_dir = {
            'audio_id': audio_id,
            'language': english_language_id,
            'audio': {
                'path': str(Path(wav_path).resolve()),
                'array': temp_array,
                'sampling_rate': 16000
            },
            'raw_text': raw_text,
            'normalized_text': raw_text,
            'gender': 'male',
            'speaker_id': 967190, # this get from all speaker_id 
            'is_gold_transcript': True,
            'accent': 'None'
         }

         data_list.append(formatted_data)
         
    data_dict = {
        'audio_id': [item['audio_id'] for item in data_list],
        'language': [item['language'] for item in data_list],
        'audio': [item['audio'] for item in data_list],
        'raw_text': [item['raw_text'] for item in data_list],
        'normalized_text': [item['normalized_text'] for item in data_list],
        'gender': [item['gender'] for item in data_list],
        'speaker_id': [item['speaker_id'] for item in data_list],
        'is_gold_transcript': [item['is_gold_transcript'] for item in data_list],
        'accent': [item['accent'] for item in data_list],
     }
     features = Features({
         'audio_id': Value('string'),
         'language': Value('int32'),
         'audio': Audio(sampling_rate=16000),
         'raw_text': Value('string'),
         'normalized_text': Value('string'),
         'gender': Value('string'),
         'speaker_id': Value('string'),
         'is_gold_transcript': Value('bool'),
         'accent': Value('string')
      })

     dataset = Dataset.from_dict(data_dict, features=features)
     #print(dataset[0])
     processed_example = prepare_dataset(dataset[20]) 
     #do the sample data for all dataset
     dataset = dataset.map(prepare_dataset_for_model, remove_columns=dataset.column_names)
     
     return dateset


def prepare_dataset_for_model(example):
    audio = example["audio"]

    example = processor(
        text=example["normalized_text"],
        audio_target=audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_attention_mask=False,
    )

    example["labels"] = example["labels"][0]
    example["speaker_embeddings"] = get_speaker_embedding(audio["array"])

    return example














