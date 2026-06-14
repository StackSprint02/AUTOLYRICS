import traceback
import torch

try:
    print("Forcing device to CPU for verification...")
    # Force cpu
    import train
    train.device = "cpu"
    
    print("Running training main on CPU...")
    train.main()
    print("CPU Training completed successfully!")
except Exception as e:
    with open("cpu_train_error.txt", "w", encoding="utf-8") as f:
        f.write(traceback.format_exc())
    print("Error occurred and written to cpu_train_error.txt")
