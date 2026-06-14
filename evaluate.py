from preprocess import load_and_prepare_dataset
import os
import time
import json
import torch
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from jiwer import wer, cer

# Configuration
MODEL_ID = "openai/whisper-tiny"
OUTPUT_DIR_DECODER = "./models/whisper-tiny-lora-decoder"
OUTPUT_DIR_BOTH = "./models/whisper-tiny-lora-both"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load processor and tokenizer
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="english", task="transcribe")

def load_fixed_model():
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    original_forward = model.forward
    def fixed_forward(*args, **kwargs):
        keys_to_pop = [k for k, v in kwargs.items() if v is None]
        for k in keys_to_pop:
            kwargs.pop(k)
        return original_forward(*args, **kwargs)
    model.forward = fixed_forward
    return model


def evaluate_model(model_name, get_model_fn, test_dataset):
    print(f"\n--- Evaluating {model_name} ---")
    
    # Clean memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
    start_vram = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    
    # Load model
    model = get_model_fn()
    model.to(device)
    model.eval()
    
    predictions = []
    references = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for sample in tqdm(test_dataset):
            # Preprocess audio features
            input_features = processor(sample["audio_array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features
            input_features = input_features.to(device)
            
            # Predict
            predicted_ids = model.generate(input_features=input_features)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            
            predictions.append(transcription.lower().strip())
            references.append(sample["text"].lower().strip())
            
    latency = time.time() - start_time
    avg_latency = latency / len(test_dataset)
    
    # Compute metrics
    # Handle empty references to avoid errors
    valid_refs = []
    valid_preds = []
    for r, p in zip(references, predictions):
        if len(r.strip()) > 0:
            valid_refs.append(r)
            valid_preds.append(p)
            
    model_wer = wer(valid_refs, valid_preds) if valid_refs else 1.0
    model_cer = cer(valid_refs, valid_preds) if valid_refs else 1.0
    
    peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    peak_vram_mb = (peak_vram - start_vram) / (1024 * 1024)
    if peak_vram_mb < 0:
        peak_vram_mb = 0
        
    print(f"WER: {model_wer:.4f} | CER: {model_cer:.4f}")
    print(f"Total Latency: {latency:.2f}s | Avg Latency: {avg_latency:.4f}s/sample")
    print(f"Peak VRAM: {peak_vram_mb:.2f} MB")
    
    # Sample predictions comparison
    print("Samples:")
    for i in range(min(3, len(valid_refs))):
        print(f"  Ref: '{valid_refs[i]}'")
        print(f"  Pred: '{valid_preds[i]}'")
        
    # Free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return {
        "wer": model_wer,
        "cer": model_cer,
        "total_latency": latency,
        "avg_latency": avg_latency,
        "peak_vram_mb": peak_vram_mb,
        "predictions": predictions[:10],
        "references": references[:10]
    }

def main():
    # Load raw dataset
    raw_dataset = load_and_prepare_dataset()
    
    # Split identically to train.py
    split_dataset = raw_dataset.train_test_split(test_size=0.2, seed=42)
    test_dataset = split_dataset["test"]
    print(f"Test size: {len(test_dataset)}")
    
    results = {}
    
    # 1. Zero-shot baseline
    def get_baseline():
        return load_fixed_model()
    results["baseline"] = evaluate_model("Zero-shot Baseline", get_baseline, test_dataset)
    
    # 2. LoRA Decoder
    def get_lora_decoder():
        base = load_fixed_model()
        if os.path.exists(OUTPUT_DIR_DECODER):
            print(f"Loading adapter from {OUTPUT_DIR_DECODER}...")
            return PeftModel.from_pretrained(base, OUTPUT_DIR_DECODER)
        else:
            print("WARNING: LoRA Decoder checkpoint not found, using baseline.")
            return base
    results["lora_decoder"] = evaluate_model("LoRA Decoder", get_lora_decoder, test_dataset)
    
    # 3. LoRA Both
    def get_lora_both():
        base = load_fixed_model()
        if os.path.exists(OUTPUT_DIR_BOTH):
            print(f"Loading adapter from {OUTPUT_DIR_BOTH}...")
            return PeftModel.from_pretrained(base, OUTPUT_DIR_BOTH)
        else:
            print("WARNING: LoRA Both checkpoint not found, using baseline.")
            return base
    results["lora_both"] = evaluate_model("LoRA Encoder+Decoder", get_lora_both, test_dataset)
    
    # Print comparison table
    print("\n" + "="*50)
    print("                 COMPARATIVE RESULTS")
    print("="*50)
    print(f"{'Approach':<25} | {'WER':<8} | {'CER':<8} | {'Avg Latency':<12} | {'VRAM (MB)':<10}")
    print("-"*70)
    for name, metrics in results.items():
        print(f"{name:<25} | {metrics['wer']:<8.4f} | {metrics['cer']:<8.4f} | {metrics['avg_latency']:<12.4f} | {metrics['peak_vram_mb']:<10.2f}")
    print("="*70)
    
    # Check relative improvement (target: >15%)
    baseline_wer = results["baseline"]["wer"]
    best_wer = min(results["lora_decoder"]["wer"], results["lora_both"]["wer"])
    if baseline_wer > 0:
        relative_improvement = (baseline_wer - best_wer) / baseline_wer * 100
        print(f"Relative WER reduction vs. Baseline: {relative_improvement:.2f}%")
        if relative_improvement >= 15.0:
            print("SUCCESS: Target of >15% relative improvement achieved!")
        else:
            print("WARNING: Relative improvement is below the target 15%.")
    else:
        print("Baseline WER is 0, cannot compute relative improvement.")
        relative_improvement = 0.0
        
    results["summary"] = {
        "relative_improvement_percent": relative_improvement,
        "success": relative_improvement >= 15.0
    }
    
    # Save results to JSON
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved evaluation results to eval_results.json")

if __name__ == "__main__":
    main()
