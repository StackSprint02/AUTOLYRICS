import sys
import traceback

def log(msg):
    print(msg)
    with open("debug_log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")

# Clear log
open("debug_log.txt", "w", encoding="utf-8").close()

try:
    log("Step 1: Importing packages...")
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
    from peft import LoraConfig, get_peft_model
    from preprocess import load_and_prepare_dataset
    from train import DataCollatorSpeechSeq2SeqWithPadding, prepare_dataset
    
    log("Step 2: Loading dataset...")
    raw_dataset = load_and_prepare_dataset()
    log(f"Raw dataset loaded: {len(raw_dataset)} samples.")
    
    log("Step 3: Loading processor...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="english", task="transcribe")
    
    log("Step 4: Mapping dataset...")
    processed_dataset = raw_dataset.map(prepare_dataset, num_proc=1)
    split_dataset = processed_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    log("Dataset splitting complete.")
    
    log("Step 5: Loading Whisper model...")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    
    log("Step 6: Wrapping with PEFT...")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, peft_config)
    
    log("Step 7: Defining training args on CPU...")
    training_args = Seq2SeqTrainingArguments(
        output_dir="./test_lora_cpu",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        max_steps=2,  # Only 2 steps for test
        predict_with_generate=True,
        fp16=False,
        report_to="none",
        remove_unused_columns=False,
    )
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    
    log("Step 8: Initializing trainer...")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
    )
    log("Trainer initialized successfully!")
    
    log("Step 9: Running trainer.train()...")
    trainer.train()
    log("Training completed successfully!")
    
except BaseException as e:
    err_msg = traceback.format_exc()
    log(f"CRASH OCCURRED:\n{err_msg}")
    raise e
