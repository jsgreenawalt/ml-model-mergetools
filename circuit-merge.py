'''
Copyright 2024 Joseph S. Greenawalt

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
import argparse
import os
import shutil
import random
import json
import gc
import math
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
import torch
#torch.use_deterministic_algorithms(True) # <- this wrecks performance, not needed anyway, apparently
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
import yaml

SEED = 255

def set_seed(seed=None):
    global SEED
    if seed == None:
        seed = SEED
    print("Setting seed to:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

MAX_CHUNK_SIZE = 5 * 1024 * 1024 * 1024  # 5GB chunk size

CONFIG_FILES = [
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenizer.json"
]


if torch.cuda.is_available():
    device = torch.device("cuda")
    DEVICE = "cuda"
else:
    device = torch.device("cpu")
    DEVICE = "cpu"
print("DEVICE=", DEVICE)
print("device=", device)
torch.set_grad_enabled(False)
torch.no_grad()

def load_tensors_from_directory(directory: str) -> Dict[str, str]:
    all_tensors = {}
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.safetensors'):
            file_path = os.path.join(directory, filename)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                all_tensors.update({k: file_path for k in f.keys()})
    return all_tensors

# utility function for OOM debugging
def print_gpu_memory_stats():
    if torch.cuda.is_available():
        # Get the current device (assuming you're using the default CUDA device)
        device = torch.cuda.current_device()

        # Get the total amount of memory in bytes
        total_memory = torch.cuda.get_device_properties(device).total_memory

        # Get the amount of memory allocated by tensors in bytes
        allocated_memory = torch.cuda.memory_allocated(device)

        # Get the amount of memory reserved by the caching allocator in bytes
        reserved_memory = torch.cuda.memory_reserved(device)

        # Calculate the amount of free memory
        free_memory = total_memory - allocated_memory - reserved_memory

        print(f"GPU Memory Usage:")
        print(f"  Total:     {total_memory / 1e9:.2f} GB")
        print(f"  Allocated: {allocated_memory / 1e9:.2f} GB")
        print(f"  Reserved:  {reserved_memory / 1e9:.2f} GB")
        print(f"  Free:      {free_memory / 1e9:.2f} GB")
    else:
        print("CUDA is not available. Running on CPU.")


# utility function for OOM debugging
def get_free_gpu_memory():
    if torch.cuda.is_available():
        # Get the current device (assuming you're using the default CUDA device)
        device = torch.cuda.current_device()

        # Get the total amount of memory in bytes
        total_memory = torch.cuda.get_device_properties(device).total_memory

        # Get the amount of memory allocated by tensors in bytes
        allocated_memory = torch.cuda.memory_allocated(device)

        # Get the amount of memory reserved by the caching allocator in bytes
        reserved_memory = torch.cuda.memory_reserved(device)

        # Calculate the amount of free memory
        #free_memory = total_memory - allocated_memory - reserved_memory
        free_memory = total_memory - allocated_memory

        print(f"GPU Memory Usage:")
        print(f"  Total:     {total_memory / 1e9:.2f} GB")
        print(f"  Allocated: {allocated_memory / 1e9:.2f} GB")
        print(f"  Reserved:  {reserved_memory / 1e9:.2f} GB")
        print(f"  Free:      {free_memory / 1e9:.2f} GB")
        return free_memory
    else:
        print("CUDA is not available. Running on CPU.")
        return None



# Memory usage of torch count_nonzero is too high
def memory_efficient_count_nonzero(tensor: torch.Tensor, chunk_multiplier: int = 10) -> int:
    # Handle scalar tensors
    if tensor.dim() == 0:
        return 1 if tensor != 0 else 0
    
    # Handle 1D tensors
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    
    total_count = 0
    num_rows = tensor.shape[0]
    chunk_size = num_rows * chunk_multiplier
    
    for i in range(0, num_rows, chunk_size):
        chunk = tensor[i:i+chunk_size]
        total_count += (chunk != 0).sum().item()
    
    return total_count


# Original version - this is my own 'reference implementation'
def magprune(delta: torch.Tensor, p: float, epsilon: float) -> torch.Tensor:
    # Rank the delta parameters based on their magnitudes
    #delta = delta.to(device) # <- should already be on device
    ranked_indices = torch.argsort(torch.abs(delta).flatten())
    flattened_delta = delta.flatten()
    
    # Calculate drop probabilities
    n = len(ranked_indices)
    drop_probs = torch.linspace(p - epsilon/2, p + epsilon/2, n, device=device)
    
    # Create a mask for dropped parameters
    mask = torch.rand(n, device=device) < drop_probs
    
    # Apply the mask and rescale
    pruned_delta = flattened_delta.clone()
    pruned_delta[ranked_indices[mask]] = 0
    pruned_delta[ranked_indices[~mask]] /= (1 - drop_probs[~mask])
    
    return pruned_delta.reshape(delta.shape)

# Memory-efficient version
def memory_efficient_magprune(delta: torch.Tensor, p: float, epsilon: float, chunk_size: int = 1000000, skip_rescaling: bool = False) -> torch.Tensor:
    device = delta.device
    original_shape = delta.shape
    n = delta.numel()

    # Estimate global statistics without materializing the entire abs(delta)
    global_max = float('-inf')
    global_min = float('inf')
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = delta.flatten()[start:end]
        chunk_abs = torch.abs(chunk)
        global_max = max(global_max, chunk_abs.max().item())
        global_min = min(global_min, chunk_abs.min().item())
        del chunk, chunk_abs

    # In-place pruning
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = delta.flatten()[start:end]

        # Approximate ranking within the chunk
        chunk_abs = torch.abs(chunk)
        chunk_ranks = 1 - ((chunk_abs - global_min) / (global_max - global_min))

        # Calculate drop probabilities for this chunk
        chunk_drop_probs = p - epsilon/2 + epsilon * chunk_ranks

        # Create a mask for dropped parameters
        chunk_mask = torch.rand(end - start, device=device) < chunk_drop_probs

        # Apply the mask and optionally rescale in-place
        chunk[chunk_mask] = 0
        if not skip_rescaling:
            chunk[~chunk_mask] /= (1 - chunk_drop_probs[~chunk_mask])

        # Free up memory
        del chunk_abs, chunk_ranks, chunk_drop_probs, chunk_mask

    return delta

'''
# just some random witchcraft
def calculate_p(i, total_count):
    desired_spread = 0.1 * (total_count - 1)  # This makes the spread increase with total_count
    x = (i - (total_count + 1) / 2) / (total_count - 1)  # Normalize i to be between -0.5 and 0.5
    return 0.5 + (desired_spread / 2) * -math.tanh(2 * x)
'''
# I'm not a witch! They dressed me up like this!
def calculate_p(i, count, temperature=0.0, base_droprate=0.5):
    if count == 1:
        return base_droprate * 2
    # calculate number of divisions
    if i == 1: i = 2
    divisions = (count - i) + 1
    # calculate our division basis multiplier
    final = 0.5 ** divisions
    # adjust final by double the inverse drop rate
    final = final * ((1 - base_droprate) * 2)
    # interpolate between geometric decay and equal drop rate at base_droprate
    final = ((1 - temperature) * final) + (temperature * (1 - base_droprate))
    # It's a fair cop.
    return 1.0 - final

'''
# Temperature-Based Model Merging Algorithm with Sign-Based Conflict Resolution

1. Initialize:
   - Set `accumulated_deltas` to a tensor of zeros, same shape as model parameters
   - Set `contribution_count` to a tensor of zeros, same shape as model parameters
   - Define `meaningful_change_threshold` (e.g., 2e-6)
   - Set `n` as the number of variant models

2. For each variant model, from lowest to highest priority:
   a. Calculate `delta` = variant_parameters - base_parameters
   b. Apply threshold: set `delta` to 0 where |delta| < meaningful_change_threshold
   c. Apply MAGPRUNE to reduce noise and sparsify deltas
   d. Identify non-zero deltas after MAGPRUNE

   e. For first-time contributions (where contribution_count == 0):
      - Set accumulated_deltas to the pruned delta
      - Set contribution_count to 1

   f. For subsequent contributions:
      - Check sign agreement between accumulated_deltas and pruned_delta
      - For sign disagreement:
        * Overwrite accumulated_deltas with pruned_delta
        * Reset contribution_count to 1
      - For sign agreement:
        * Increment contribution_count
        * Calculate exponential decay average: (accumulated_deltas + pruned_delta) / 2
        * Calculate true running average: (accumulated_deltas * (count - 1) + pruned_delta) / count
        * Update accumulated_deltas:
          accumulated_deltas = (1 - temperature) * exp_decay_avg + temperature * true_running_avg

3. Final merged parameters = base_parameters + accumulated_deltas

Key Behaviors:
- First contributions are always accepted as-is
- Subsequent contributions:
  - If signs disagree, the new contribution overwrites the existing one
  - If signs agree, the contribution is merged using the temperature-based algorithm
- At temperature 0:
  - Each new contribution with sign agreement takes 50% influence, halving the influence of previous accumulated contributions
  - Previous contributions are collectively reduced to 50% of their prior influence with each contribution
- At temperature 1:
  - Contributions with sign agreement are weighted according to a true running average
- Temperatures between 0 and 1 interpolate between these behaviors
- Zero deltas (below threshold or pruned by MAGPRUNE) are ignored and don't affect weighting
- MAGPRUNE is applied to reduce noise and sparsify deltas before merging
'''
def process_and_merge_tensor(base_file: str, variant_files: List[str], tensor_name: str, temperature: float, base_droprate: float, meaningful_change_threshold: float, skip_rescaling: bool = False) -> torch.Tensor:
    global device
    print(f"Processing tensor: {tensor_name}")
    gc.collect()
    torch.cuda.empty_cache()

    # Create a new CUDA stream
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with safe_open(base_file, framework="pt", device="cpu") as base_f:
            base_tensor = base_f.get_tensor(tensor_name).to(device) #.half()

        shape = base_tensor.shape
        dtype = base_tensor.dtype
        device = torch.device("cuda")

        #accumulated_deltas = torch.zeros_like(base_tensor, dtype=torch.float16, device=device)
        accumulated_deltas = torch.zeros_like(base_tensor, dtype=torch.float32, device=device)
        contribution_count = torch.zeros_like(base_tensor, dtype=torch.int8, device=device)

        # Dynamic batch size adjustment - go as fast as we can without running out of vram
        free_memory = get_free_gpu_memory()
        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        try: row_size_bytes = shape[1] * bytes_per_element
        except: row_size_bytes = bytes_per_element
        print("free memory:", free_memory)
        memory_usage_factor = 12.0  # Adjust upwards until memory errors go away, lazy, LOL 
        rows_per_chunk = int(free_memory / (row_size_bytes * len(variant_files) * memory_usage_factor))
        rows_per_chunk = max(1, rows_per_chunk)
        print("shape=", shape)
        print("shape=", row_size_bytes)
        print("rows_per_chunk=", rows_per_chunk)

        print("Drop rates for models (p)")
        #for i in range(1, len(variant_files)+1):
        for i, variant_file in enumerate(reversed(variant_files), 1):
            p = calculate_p(i, len(variant_files), temperature, base_droprate)
            print(f"i={i}, p={p} variant_file={variant_file}")

        for i, variant_file in enumerate(reversed(variant_files), 1):
            with safe_open(variant_file, framework="pt", device="cpu") as variant_f:
                for start_row in range(0, shape[0], rows_per_chunk):
                    end_row = min(start_row + rows_per_chunk, shape[0])

                    base_chunk = base_tensor[start_row:end_row]
                    variant_chunk = variant_f.get_tensor(tensor_name)[start_row:end_row].to(device) #.half()

                    delta = variant_chunk - base_chunk
                    del variant_chunk
                    gc.collect()
                    torch.cuda.empty_cache()

                    significant_mask = torch.abs(delta) >= meaningful_change_threshold
                    delta[~significant_mask] = 0

                    non_zero_count = memory_efficient_count_nonzero(delta)
                    if non_zero_count == 0:
                        print(f"   No meaningful changes found in rows {start_row} to {end_row} for this variant.")
                        del delta, significant_mask
                        torch.cuda.empty_cache()
                        continue

                    p = calculate_p(i, len(variant_files), temperature, base_droprate)
                    #print(f"i={i}, p={p}")
                    if p == 0.0:
                        # skip pruning if we don't need to
                        #print("Pruning skipped for p=0.0, yay?")
                        pruned_delta = delta
                    else:
                        pruned_delta = memory_efficient_magprune(delta, p=p, epsilon=0.1, skip_rescaling=skip_rescaling)
                    del delta
                    gc.collect()
                    torch.cuda.empty_cache()

                    non_zero_mask = pruned_delta != 0
                    if torch.any(non_zero_mask):
                        acc_chunk = accumulated_deltas[start_row:end_row]
                        count_chunk = contribution_count[start_row:end_row]

                        first_contribution_mask = count_chunk == 0
                        acc_chunk[first_contribution_mask & non_zero_mask] = pruned_delta[first_contribution_mask & non_zero_mask]
                        count_chunk[first_contribution_mask & non_zero_mask] = 1

                        subsequent_mask = ~first_contribution_mask & non_zero_mask
                        if torch.any(subsequent_mask):
                            sign_agreement = (torch.sign(acc_chunk) == torch.sign(pruned_delta)) & subsequent_mask
                            sign_disagreement = ~sign_agreement & subsequent_mask

                            acc_chunk[sign_disagreement] = pruned_delta[sign_disagreement]
                            count_chunk[sign_disagreement] = 1

                            if torch.any(sign_agreement):
                                count_chunk[sign_agreement] += 1
                                exp_decay_avg = (acc_chunk[sign_agreement] + pruned_delta[sign_agreement]) / 2
                                true_running_avg = (acc_chunk[sign_agreement] * (count_chunk[sign_agreement] - 1) + pruned_delta[sign_agreement]) / count_chunk[sign_agreement]
                                acc_chunk[sign_agreement] = (1 - temperature) * exp_decay_avg + temperature * true_running_avg

                        accumulated_deltas[start_row:end_row] = acc_chunk
                        contribution_count[start_row:end_row] = count_chunk

                    del pruned_delta, non_zero_mask
                    gc.collect()
                    torch.cuda.empty_cache()
                    stream.synchronize()

            print(f"   Tensor deltas have been accumulated for variant.")

        merged_tensor = (base_tensor + accumulated_deltas) #.half()

        try: del base_tensor
        except: pass
        try: del accumulated_deltas
        except: pass
        try: del contribution_count
        except: pass
        gc.collect()
        stream.synchronize()
        torch.cuda.empty_cache()

    # Ensure all operations in the stream are completed and clean up
    gc.collect()
    stream.synchronize()
    torch.cuda.empty_cache()

    return merged_tensor

def save_chunked_safetensors(state_dict: Dict[str, torch.Tensor], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    current_chunk = {}
    current_chunk_size = 0
    chunk_number = 1
    total_chunks = 0

    # First pass to determine total number of chunks
    temp_size = 0
    for tensor in state_dict.values():
        temp_size += tensor.numel() * tensor.element_size()
        if temp_size >= MAX_CHUNK_SIZE:
            total_chunks += 1
            temp_size = 0
    if temp_size > 0:
        total_chunks += 1

    index_json = {"metadata": {}, "weight_map": {}}

    for key, tensor in tqdm(state_dict.items(), desc="Saving chunks"):
        tensor_size = tensor.numel() * tensor.element_size()

        if current_chunk_size + tensor_size > MAX_CHUNK_SIZE:
            # Save current chunk
            chunk_filename = f"model-{chunk_number:05d}-of-{total_chunks:05d}.safetensors"
            save_file(current_chunk, os.path.join(output_dir, chunk_filename), metadata={"format": "pt"})
            print(f"Saved {chunk_filename}")

            # Update index.json
            for tensor_key in current_chunk.keys():
                index_json["weight_map"][tensor_key] = chunk_filename

            # Reset for next chunk
            current_chunk = {}
            current_chunk_size = 0
            chunk_number += 1

        current_chunk[key] = tensor
        current_chunk_size += tensor_size

    # Save last chunk if not empty
    if current_chunk:
        chunk_filename = f"model-{chunk_number:05d}-of-{total_chunks:05d}.safetensors"
        save_file(current_chunk, os.path.join(output_dir, chunk_filename), metadata={"format": "pt"})
        print(f"Saved {chunk_filename}")

        # Update index.json for the last chunk
        for tensor_key in current_chunk.keys():
            index_json["weight_map"][tensor_key] = chunk_filename

    # Save index.json
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index_json, f, indent=2)

def copy_config_files(base_dir: str, output_dir: str):
    for file in CONFIG_FILES:
        src = os.path.join(base_dir, file)
        dst = os.path.join(output_dir, file)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"Copied {file} to output directory")
        else:
            print(f"Warning: {file} not found in base model directory")

def merge_models(base_dir: str, variant_dirs: List[str], output_dir: str, temperature: float, base_droprate: float, meaningful_change_threshold: float, skip_rescaling: bool = False):
    global SEED
    print(f"Base directory: {base_dir}")
    for i, variant_dir in enumerate(variant_dirs, 1):
        print(f"Variant {i} directory: {variant_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Temperature: {temperature}")

    print("Loading tensor metadata...")
    base_tensors = load_tensors_from_directory(base_dir)
    variant_tensors = [load_tensors_from_directory(variant_dir) for variant_dir in variant_dirs]

    all_tensor_names = set(base_tensors.keys())
    for variant_tensor in variant_tensors:
        all_tensor_names.update(variant_tensor.keys())

    print(f"Total unique tensors: {len(all_tensor_names)}")

    print("Merging tensors:")
    merged_state_dict = {}
    all_tensor_names = list(all_tensor_names)
    all_tensor_names.sort()
    i = 0
    for tensor_name in tqdm(all_tensor_names):
        i += 1
        set_seed(SEED) # <- this, here specifically in this place, saves us from non-determinism, but perhaps only between runs on *this* machine
        set_seed(SEED + i * random.randint(0, 65535)) # <- We don't want the same seed pattern on every tensor, but this is *based* on the above seed and deterministic
        base_file = base_tensors.get(tensor_name)
        if base_file is None:
            base_file = next((variant[tensor_name] for variant in variant_tensors if tensor_name in variant), None)

        if base_file is None:
            print(f"Warning: Tensor {tensor_name} not found in any model. Skipping.")
            continue

        variant_files = [variant[tensor_name] for variant in variant_tensors if tensor_name in variant]

        merged_tensor = process_and_merge_tensor(base_file, variant_files, tensor_name, temperature, base_droprate, meaningful_change_threshold, skip_rescaling=skip_rescaling)
        merged_state_dict[tensor_name] = merged_tensor.cpu()  # Move the tensor back to CPU
        del merged_tensor
        gc.collect()
        torch.cuda.empty_cache()

    print("Saving merged model as chunked SafeTensors files...")
    save_chunked_safetensors(merged_state_dict, output_dir)

    print("Copying configuration files...")
    #copy_config_files(base_dir, output_dir)

    print(f"Merged model and configuration files saved to {output_dir}")

def load_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_synthetic_data(num_blocks=3, rows=10, cols=10):
    return {f"tensor_{i}": torch.rand(rows, cols) for i in range(num_blocks)}

def perturb_model(model, perturbation_rate=0.5):
    perturbed = {}
    total_params = 0
    total_perturbed = 0
    
    for name, tensor in model.items():
        # Create the mask for perturbation
        mask = torch.rand_like(tensor) < perturbation_rate
        
        # Create quantized perturbation values
        quantized_perturbation = torch.randint(0, 3, tensor.shape, device=tensor.device).float() - 1
        quantized_perturbation *= 0.01
        
        # Apply the perturbation
        perturbed[name] = tensor + mask * quantized_perturbation

        total_params += tensor.numel()
        total_perturbed += mask.sum().item()

    print(f"Total parameters: {total_params}")
    print(f"Perturbed parameters: {total_perturbed}")
    print(f"Perturbation percentage: {total_perturbed / total_params * 100:.2f}%")
    
    return perturbed


def save_model(model, directory):
    os.makedirs(directory, exist_ok=True)
    save_file(model, os.path.join(directory, "model.safetensors"))

def print_models(base, variant1, variant2, merged):
    for name in base.keys():
        print(f"\nTensor: {name}")
        print("Base Model:".ljust(80) + "| " + "Variant 1:".ljust(80) + "| " + "Variant 2:".ljust(80) + "| " + "Merged Model:")
        print("-" * 110)
        for i in range(base[name].shape[0]):
            base_row = " ".join(f"{x:7.4f}" for x in base[name][i])
            var1_row = " ".join(f"{x:7.4f}" for x in variant1[name][i])
            var2_row = " ".join(f"{x:7.4f}" for x in variant2[name][i])
            merged_row = " ".join(f"{x:7.4f}" for x in merged[name][i])
            print(f"{base_row} | {var1_row} | {var2_row} | {merged_row}")

def print_models(base, variant1, variant2, merged):
    # ANSI color codes
    RESET = "\033[0m"
    GREEN = "\033[32m"
    BLUE = "\033[34m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"

    def color_value(base_val, var1_val, var2_val, merged_val, model_index):
        # Helper function to check if values are close
        def is_close(a, b):
            return np.isclose(a, b, rtol=5e-5, atol=5e-5)

        # Priority 1: Param same across all models
        if is_close(base_val, var1_val) and is_close(base_val, var2_val) and is_close(base_val, merged_val):
            return GREEN

        # Priority 2: Individual param same as base param
        if model_index == 0 or is_close(base_val, [var1_val, var2_val, merged_val][model_index - 1]):
            return BLUE

        # Priority 3: Param same as another model, but not base
        if model_index > 1:
            if is_close(var1_val, var2_val) and model_index in [2, 3]:
                return YELLOW
            if model_index == 3 and is_close(merged_val, var2_val):
                return MAGENTA
            if model_index == 3 and is_close(merged_val, var1_val):
                return YELLOW

        # Priority 4: Param is unique for that position
        model_colors = [BLUE, YELLOW, MAGENTA, CYAN]
        return model_colors[model_index]

    for name in base.keys():
        print(f"\n{CYAN}Tensor: {name}{RESET}")
        print(f"{BLUE}Base Model:".ljust(80) + "| " + f"{YELLOW}Variant 1:".ljust(80) + "| " + f"{MAGENTA}Variant 2:".ljust(80) + f"| {CYAN}Merged Model:{RESET}")
        print("-" * 110)

        for i in range(base[name].shape[0]):
            base_row = base[name][i]
            var1_row = variant1[name][i]
            var2_row = variant2[name][i]
            merged_row = merged[name][i]

            for model_index, row in enumerate([base_row, var1_row, var2_row, merged_row]):
                for j in range(len(row)):
                    color = color_value(base_row[j], var1_row[j], var2_row[j], merged_row[j], model_index)
                    print(f"{color}{row[j]:7.5f}{RESET}", end=" ")
                print("|", end=" ")
            print()


def run_test(temperature, seed):
    #torch.use_deterministic_algorithms(True) # <- kills performance, not guaranteed, and seemingly not needed on my machine
    set_seed(seed)

    # Create synthetic data
    base_model = create_synthetic_data()
    variant1 = perturb_model(base_model)
    variant2 = perturb_model(base_model)
    #variant1 = base_model
    #variant2 = variant1

    # Save models to temporary directories
    temp_dir = "./temp_test_models"
    os.makedirs(temp_dir, exist_ok=True)
    save_model(base_model, os.path.join(temp_dir, "base"))
    save_model(variant1, os.path.join(temp_dir, "variant1"))
    save_model(variant2, os.path.join(temp_dir, "variant2"))

    # Run merge_models function
    output_dir = os.path.join(temp_dir, "output")
    merge_models(
        os.path.join(temp_dir, "base"),
        [os.path.join(temp_dir, "variant1"), os.path.join(temp_dir, "variant2")],
        output_dir,
        temperature,
        skip_rescaling=True
    )

    # Load merged model
    with safe_open(os.path.join(output_dir, "model-00001-of-00001.safetensors"), framework="pt", device="cpu") as f:
        merged_model = {k: f.get_tensor(k) for k in f.keys()}

    # Print results
    print_models(base_model, variant1, variant2, merged_model)

    # Clean up temporary directories
    shutil.rmtree(temp_dir)

def main():
    global SEED
    parser = argparse.ArgumentParser(description="Merge a base SafeTensors model with multiple variant models.")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("base_model", type=str, nargs='?', help="Base SafeTensors model directory")
    parser.add_argument("variant_models", type=str, nargs='*', help="Variant SafeTensors model directories")
    parser.add_argument("output_dir", type=str, nargs='?', help="Output directory for the merged model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature parameter interpolate between geometric decay and true average for drop rate and weight merge (default: 0.0)")
    parser.add_argument("--base_droprate", type=float, default=0.5, help="Base drop-rate set initial p for highest priority model (default: 0.5)")
    parser.add_argument("--meaningful_change_threshold", type=float, default=2e-6, help="Consider any weight delta below this value to be zero, no change (default: 2e-6)")
    parser.add_argument("--seed", type=int, default=0, help="Set a fixed seed for torch and python rng")
    parser.add_argument("--test", action="store_true", help="Run test function")
    parser.add_argument("--skip_rescaling", action="store_true", help="Skip rescaling of weights after pruning")
    args = parser.parse_args()
    print(args)
    if args.test:
        run_test(args.temperature, args.seed)
    else:
        # Existing code for merging models
        if args.config:
            config = load_config(args.config)
            args.base_model = config['base_model']
            args.variant_models = [variant['model'] for variant in config['variant_models']]
            args.output_dir = config['output_dir']
            if 'parameters' in config:
                setattr(args, "base_droprate", 0.5) # default 0.5
                setattr(args, "meaningful_change_threshold", 2e-6) # default 2e-6
                for key, value in config['parameters'].items():
                    setattr(args, key, value)
        else:
            if not args.base_model or not args.variant_models or not args.output_dir:
                parser.print_help()
                sys.exit(1)
        
        SEED = args.seed
        set_seed(args.seed)

        merge_models(args.base_model, args.variant_models, args.output_dir, args.temperature, args.base_droprate, args.meaningful_change_threshold, skip_rescaling=args.skip_rescaling)

if __name__ == "__main__":
    main()
