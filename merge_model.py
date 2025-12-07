from safetensors.torch import load_file, save_file
import glob, sys, os

# Get input from cli
model_dir = sys.argv[1]
output_file = os.path.join(model_dir, "model.safetensors")

state_dict = {}

print(f"Merging models from {model_dir} into {output_file}")

for f in sorted(glob.glob(os.path.join(model_dir, "model-*-of-*.safetensors"))):
    tensors = load_file(f)
    state_dict.update(tensors)

print(f"Saving merged model to {output_file}")
save_file(state_dict, output_file)