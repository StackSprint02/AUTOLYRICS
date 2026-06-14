import torch
from transformers import WhisperForConditionalGeneration
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

# Wrap the Whisper model forward method to pop all None keyword arguments
original_model_forward = model.forward

def fixed_model_forward(*args, **kwargs):
    # Pop any None keyword arguments to avoid duplicates in submodules (like WhisperDecoder)
    keys_to_pop = [k for k, v in kwargs.items() if v is None]
    for k in keys_to_pop:
        kwargs.pop(k)
    return original_model_forward(*args, **kwargs)

model.forward = fixed_model_forward

# Wrap with PEFT
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, peft_config)

# Run a dummy forward pass
input_features = torch.randn(1, 80, 3000)
labels = torch.tensor([[50258, 50259, 50363]])
try:
    outputs = model(input_features=input_features, labels=labels)
    print("Forward pass succeeded!")
except Exception as e:
    import traceback
    traceback.print_exc()
