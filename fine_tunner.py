import argparse
from transformers import Trainer, TrainingArguments, SpeechT5ForConditionalGeneration, SpeechT5Processor
from datasets import load_dataset
from functools import partial
from data_prosessiong import get_dataset

def fine_tune_speachT5():
    #set the arguments
    parser = argparse.ArgumentParser(description="Fine-tune SpeechT5 model")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="leanning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type=int, default=3, help="epch")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="decay weight")
    parser.add_argument("--logging_steps", type=int, default=10, help="log steps")
    parser.add_argument("--save_steps", type=int, default=500, help="save step")
    parser.add_argument("--output_dir", type=str, default="./results", help="output dir")
    args = parser.parse_args()

    #load pre-train model SpeechT5
    model = SpeechT5ForConditionalGeneration.from_pretrained("microsoft/speecht5_tts")
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint) 
   
    model.config.use_cache = False
    model.generate = partial(model.generate, use_cache=True)
    
    dataset = get_dataset( "" , )
    
    #inference data 
    inference_dataset = dataset.select(range(50))

    remaining_dataset = dataset.select(range(50, len(dataset)))
    split_dataset = remaining_dataset.train_test_split(test_size=0.1)

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir= args.output_dir + "speecht5_finetuned_en64_lr401",  # change to a repo name of your choice
        per_device_train_batch_size=args.batch_size ,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate , # le-3 le-4 le-5
        warmup_steps=500,
        max_steps=1500,  # 1500 
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=2,
        save_steps=args.save_steps ,
        eval_steps=100,
        logging_steps=args.logging_steps, # 10
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        greater_is_better=False,
        label_names=["labels"],
        push_to_hub=True,
    )  

    #idefinde Trainer 
    trainer = Trainer(
        model=model,                                 
        args=training_args,                          
        train_dataset= split_dataset['train'],           
        eval_dataset= split_dataset['text'],            
    )

    # start to train 
    trainer.train()

    #trainer.save_model("./fine_tuned_model")


if __name__ == "__main__":
    from google.colab import drive
    drive.mount('/content/drive')
   
    from huggingface_hub import notebook_login
    notebook_login()

    main()


