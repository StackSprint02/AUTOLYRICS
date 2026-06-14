import sys
import traceback

try:
    from preprocess import load_and_prepare_dataset
    import os
    import torch
    from torch.utils.data import DataLoader
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import LoraConfig, get_peft_model
    import numpy as np
except BaseException as e:
    with open("train_import_error.txt", "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())
    print("Error occurred during imports and written to train_import_error.txt")
    sys.exit(1)

# Configuration
MODEL_ID = "openai/whisper-tiny"
OUTPUT_DIR_DECODER = "./models/whisper-tiny-lora-decoder"
OUTPUT_DIR_BOTH = "./models/whisper-tiny-lora-both"

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Global processor variables to be initialized lazily
processor = None
tokenizer = None
feature_extractor = None

def init_processor():
    global processor, tokenizer, feature_extractor
    if processor is None:
        processor = WhisperProcessor.from_pretrained(MODEL_ID, language="english", task="transcribe")
        tokenizer = processor.tokenizer
        feature_extractor = processor.feature_extractor

def collate_fn(batch):
    init_processor()
    # Prepare batch of input features and labels
    input_features = [torch.tensor(item["input_features"]) for item in batch]
    labels = [torch.tensor(item["labels"]) for item in batch]
    
    # Pad input features to standard shape
    input_features = torch.stack(input_features)
    
    # Pad labels with -100 (ignored in loss)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    
    return {
        "input_features": input_features,
        "labels": padded_labels
    }

def prepare_dataset(sample):
    init_processor()
    # compute log-Mel input features
    sample["input_features"] = feature_extractor(sample["audio_array"], sampling_rate=sample["sampling_rate"]).input_features[0]
    # encode target text to label ids
    sample["labels"] = tokenizer(sample["text"]).input_ids
    return sample

def train_lora_variant(target_modules, output_dir, train_dataset, val_dataset):
    print(f"\n--- Fine-Tuning LoRA variant saving to: {output_dir} ---")
    print(f"Target Modules: {target_modules}")
    
    # Load pretrained model
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.use_cache = False
    
    # Fix PEFT duplicate inputs_embeds / input_ids argument bug on newer transformers versions
    original_forward = model.forward
    def fixed_forward(*args, **kwargs):
        keys_to_pop = [k for k, v in kwargs.items() if v is None]
        for k in keys_to_pop:
            kwargs.pop(k)
        return original_forward(*args, **kwargs)
    model.forward = fixed_forward

    
    # PEFT/LoRA config
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_2_SEQ_LM"
    )
    
    # Wrap model with PEFT
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Training Loop
    num_epochs = 10
    print(f"Starting training on {device} for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (step + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | Step {step+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        # Validation Loss
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_features = batch["input_features"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_features=input_features, labels=labels)
                total_val_loss += outputs.loss.item()
                
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Complete | Avg Train Loss: {avg_train_loss:.4f} | Avg Val Loss: {avg_val_loss:.4f}")
        
    # Save the adapter weights
    model.save_pretrained(output_dir)
    print(f"Successfully saved model checkpoint to {output_dir}")

def main():
    # Load raw dataset
    raw_dataset = load_and_prepare_dataset()
    
    # Prepare features
    print("Preparing dataset features...")
    processed_dataset = raw_dataset.map(prepare_dataset)
    
    # Train-test split (80-20)
    split_dataset = processed_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]
    
    print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}")
    
    # Initialize processor globally in main thread once
    init_processor()
    
    # Approach 1: LoRA on Decoder Only
    # Dynamically find decoder target modules to prevent PEFT naming mismatches on Windows/PEFT versions
    temp_model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    decoder_only_targets = []
    for name, _ in temp_model.named_modules():
        if "decoder" in name and (name.endswith("q_proj") or name.endswith("v_proj")):
            parts = name.split(".")
            if "decoder" in parts:
                idx = parts.index("decoder")
                decoder_only_targets.append(".".join(parts[idx:]))
    del temp_model
    print(f"Dynamically identified decoder target modules: {decoder_only_targets}")
    train_lora_variant(decoder_only_targets, OUTPUT_DIR_DECODER, train_dataset, val_dataset)
    
    # Approach 2: LoRA on Both Encoder and Decoder
    both_targets = ["q_proj", "v_proj"]
    train_lora_variant(both_targets, OUTPUT_DIR_BOTH, train_dataset, val_dataset)

if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        with open("train_error.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print("Error occurred and written to train_error.txt")
        raise e
