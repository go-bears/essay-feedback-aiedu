# train.py
import sys
from src.train import train_loop  # or wherever your logic lives

if __name__ == "__main__":
    # sys.argv: [ "train.py", run_folder, model_handle, output_model_name ]
    _, run_folder, model_handle, output_model_name = sys.argv
    # copy in the logic from train_unsloth → train_loop
    train_loop(run_folder, model_handle, output_model_name)
