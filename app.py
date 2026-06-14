from preprocess import preprocess_audio
import os
import torch
import torchaudio
import gradio as gr
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

# Configuration
MODEL_ID = "openai/whisper-tiny"
OUTPUT_DIR_DECODER = "./models/whisper-tiny-lora-decoder"
OUTPUT_DIR_BOTH = "./models/whisper-tiny-lora-both"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Gradio App using device: {device}")

# Load models and processors
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


print("Loading baseline model...")
base_model = load_fixed_model().to(device)

print("Checking for LoRA adapters...")
lora_decoder_model = None
if os.path.exists(OUTPUT_DIR_DECODER):
    try:
        print("Loading LoRA Decoder adapter...")
        lora_decoder_model = PeftModel.from_pretrained(
            load_fixed_model(), 
            OUTPUT_DIR_DECODER
        ).to(device)
    except Exception as e:
        print(f"Error loading LoRA Decoder adapter: {e}")

lora_both_model = None
if os.path.exists(OUTPUT_DIR_BOTH):
    try:
        print("Loading LoRA Encoder+Decoder adapter...")
        lora_both_model = PeftModel.from_pretrained(
            load_fixed_model(), 
            OUTPUT_DIR_BOTH
        ).to(device)
    except Exception as e:
        print(f"Error loading LoRA Encoder+Decoder adapter: {e}")

def transcribe_audio(audio_path):
    if audio_path is None:
        return "Please upload or record audio.", "N/A", "N/A", None
        
    try:
        # Load audio using torchaudio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Apply preprocessing pipeline (bandpass + RMS normalization)
        processed_wf, processed_sr = preprocess_audio(waveform, sample_rate)
        
        # Save preprocessed audio to a temporary file for playback
        temp_processed_path = "temp_preprocessed.wav"
        torchaudio.save(temp_processed_path, processed_wf, processed_sr)
        
        # Extract features for Whisper
        input_features = processor(
            processed_wf.squeeze(0).numpy(), 
            sampling_rate=processed_sr, 
            return_tensors="pt"
        ).input_features.to(device)
        
        # 1. Baseline transcription
        predicted_ids_base = base_model.generate(input_features=input_features)
        trans_base = processor.batch_decode(predicted_ids_base, skip_special_tokens=True)[0]
        
        # 2. LoRA Decoder transcription
        if lora_decoder_model is not None:
            predicted_ids_dec = lora_decoder_model.generate(input_features=input_features)
            trans_dec = processor.batch_decode(predicted_ids_dec, skip_special_tokens=True)[0]
        else:
            # Fallback/simulation for demonstration if not trained
            trans_dec = "[Adapter Not Found] " + trans_base
            
        # 3. LoRA Both transcription
        if lora_both_model is not None:
            predicted_ids_both = lora_both_model.generate(input_features=input_features)
            trans_both = processor.batch_decode(predicted_ids_both, skip_special_tokens=True)[0]
        else:
            # Fallback/simulation for demonstration if not trained
            trans_both = "[Adapter Not Found] " + trans_base
            
        return trans_base, trans_dec, trans_both, temp_processed_path
        
    except Exception as e:
        return f"Error during processing: {str(e)}", "Error", "Error", None

# Build UI
with gr.Blocks(title="AUTOLYRICS: Automatic Lyric Transcriber") as demo:
    gr.Markdown(
        """
        # 🎵 AUTOLYRICS
        ### Automatic Lyric Transcription & Model Benchmarking
        
        This application compares a standard **Zero-shot Whisper-Tiny** baseline with **LoRA (Low-Rank Adaptation)** fine-tuned versions trained on polyphonic singing audio.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 🎙️ Input Singing Audio")
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Record or Upload Song Clip")
            transcribe_btn = gr.Button("Transcribe & Compare", variant="primary")
            
            gr.Markdown("### 🔊 Preprocessed Audio")
            gr.Markdown("_This audio is processed via our DSP pipeline (150Hz-8kHz bandpass + RMS normalization) to suppress instrumentals._")
            audio_output_processed = gr.Audio(label="Processed Audio (Vocal Isolated)", interactive=False)
            
        with gr.Column(scale=1):
            gr.Markdown("### 📝 Transcription Results")
            
            with gr.Box() if hasattr(gr, "Box") else gr.Group():
                gr.Markdown("#### 🟥 1. Zero-shot Whisper-Tiny (Baseline)")
                text_base = gr.Textbox(placeholder="Baseline transcription will appear here...", label="Baseline Transcript")
                
            with gr.Box() if hasattr(gr, "Box") else gr.Group():
                gr.Markdown("#### 🟨 2. LoRA Decoder-Tuned Whisper-Tiny")
                text_dec = gr.Textbox(placeholder="LoRA Decoder transcription will appear here...", label="LoRA Decoder Transcript")
                
            with gr.Box() if hasattr(gr, "Box") else gr.Group():
                gr.Markdown("#### 🟩 3. LoRA Encoder+Decoder-Tuned Whisper-Tiny")
                text_both = gr.Textbox(placeholder="LoRA Both transcription will appear here...", label="LoRA Encoder+Decoder Transcript")
                
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=audio_input,
        outputs=[text_base, text_dec, text_both, audio_output_processed]
    )
    
    gr.Markdown(
        """
        ---
        **How it works:**
        1. **Resampling & Mono Conversion**: Standardizes raw user uploads to 16kHz mono.
        2. **Bandpass Filtering**: Suppresses sub-bass (kick/bass guitar) and high-frequency cymbals.
        3. **RMS Normalization**: Adjusts gain to optimal Whisper scale.
        4. **LoRA Transcription**: The preprocessed representation is passed through our custom low-rank adapted transformer model.
        """
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
