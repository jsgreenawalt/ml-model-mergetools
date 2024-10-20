# ml-model-mergetools

Circuit-Merge! and other tools for working with, validating, and merging Large Language Models

# Circuit-Merge! Documentation

## Overview

Circuit-Merge! is a Python script for merging multiple Language Model (LM) checkpoints using a novel approach inspired by the DELLA (Drop and rEscaLe via sampLing with mAgnitude) merging technique. This tool is designed to combine the capabilities of multiple fine-tuned models into a single model without additional training.

## Key Features

- Merges multiple LM checkpoints (base model + variants)
- Implements a temperature-based merging algorithm with sign-based conflict resolution
- Uses MAGPRUNE (Magnitude-based Pruning) to reduce interference and sparsify delta parameters
- Supports chunked SafeTensors file format for efficient storage and loading
- Provides options for customizing the merging process (e.g., temperature, drop rate, meaningful change threshold)

## Installation

### Prerequisites

- Python 3.7+
- PyTorch
- safetensors
- tqdm
- PyYAML

### Setup

1. Clone the repository or download the `circuit-merge.py` script.
2. Install the required dependencies:

```bash
pip install torch safetensors tqdm pyyaml
```

## Usage

### Command Line Interface

```bash
python circuit-merge.py [--config CONFIG_FILE] [--temperature TEMP] [--base_droprate RATE] [--meaningful_change_threshold THRESHOLD] [--seed SEED] [--skip_rescaling] BASE_MODEL VARIANT_MODEL1 [VARIANT_MODEL2 ...] OUTPUT_DIR
```

### Arguments

- `BASE_MODEL`: Directory containing the base SafeTensors model
- `VARIANT_MODEL1`, `VARIANT_MODEL2`, etc.: Directories containing variant SafeTensors models
- `OUTPUT_DIR`: Directory to save the merged model
- `--config`: Path to a YAML configuration file (optional)
- `--temperature`: Temperature parameter for interpolating between geometric decay and true average (default: 0.0)
- `--base_droprate`: Base drop rate for the highest priority model (default: 0.5)
- `--meaningful_change_threshold`: Threshold for considering weight deltas as meaningful changes (default: 2e-6)
- `--seed`: Set a fixed seed for reproducibility
- `--skip_rescaling`: Skip rescaling of weights after pruning
- `--test`: Run a test function with synthetic data

### Configuration File

You can use a YAML configuration file to specify the merge parameters:

```yaml
base_model: "/path/to/base/model"
variant_models:
  - model: "/path/to/variant1"
  - model: "/path/to/variant2"
output_dir: "/path/to/output"
parameters:
  temperature: 0.2
  base_droprate: 0.6
  meaningful_change_threshold: 1e-6
  seed: 42
```

## How It Works

Circuit-Merge! implements a temperature-based model merging algorithm with sign-based conflict resolution. The key steps in the process are:

1. **Delta Calculation**: Compute the difference (delta) between each variant model and the base model.
2. **Thresholding**: Apply a threshold to ignore insignificant changes in the delta parameters.
3. **MAGPRUNE**: Use magnitude-based pruning to reduce noise and sparsify the delta parameters.
4. **Merging**: Combine the pruned deltas using a temperature-based algorithm:
   - For first-time contributions, accept the delta as-is.
   - For subsequent contributions:
     - If signs disagree, overwrite with the new contribution.
     - If signs agree, use a weighted average based on the temperature parameter.
5. **Final Merge**: Add the accumulated deltas to the base model parameters.

### Model Priority and Processing Order

Circuit-Merge! introduces the concept of model priority to control the influence of each variant model on the final merged result. The priority is determined by the order in which variant models are listed in the YAML configuration file or command-line arguments.

Importantly, the script processes variant models in reverse order of their listing. This means that the last variant model listed in the YAML configuration is processed first and considered the lowest priority model. Conversely, the first variant model listed is processed last and has the highest priority. This processing order ensures that higher priority models have a greater influence on the final merged model.

The priority affects the merging process in several ways:

1. **Sign Determination**: When there's a sign conflict between accumulated deltas and new contributions, the sign of the higher priority model takes precedence.

2. **Weight Influence**: The dynamic drop rate calculation (discussed below) assigns lower drop rates to higher priority models, allowing them to retain more of their unique parameters.

3. **Overwriting**: In cases of sign disagreement, the contribution from a higher priority model will overwrite the existing accumulated deltas.

### Dynamic Drop Rate Calculation

The script uses a dynamic method to calculate drop rates for each variant model. This is implemented in the `calculate_p` function:

```python
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
```

Key aspects of this function:

- `i`: The reversed index of the current model (1 for highest priority, `count` for lowest)
- `count`: Total number of variant models
- `temperature`: Controls interpolation between geometric decay and equal drop rates
- `base_droprate`: The base drop rate for calculating the final rate

The function calculates the drop rate using a combination of geometric decay and temperature-based interpolation:

1. It first calculates a geometric decay based on the model's priority.
2. This decay is then adjusted based on the `base_droprate`.
3. Finally, it interpolates between this adjusted decay and a flat rate based on the temperature.

This approach ensures that:
- Higher priority models (lower `i`) have lower drop rates, preserving more of their unique parameters.
- The `temperature` parameter allows fine-tuning between a priority-based approach (low temperature) and a more democratic approach (high temperature).
- The `base_droprate` provides an overall control of how aggressive the pruning should be across all models.

### Merging Methodology

The actual merging process, implemented in `process_and_merge_tensor`, follows these steps for each tensor:

1. **Initialization**: Create tensors for accumulated deltas and contribution counts.

2. **Iterative Merging**: For each variant model, starting from the lowest priority:
   a. Calculate the delta from the base model.
   b. Apply the meaningful change threshold.
   c. Apply MAGPRUNE using the dynamically calculated drop rate.
   d. For each non-zero delta after pruning:
      - If it's a first-time contribution, accept it as-is.
      - For subsequent contributions:
        * Check sign agreement with accumulated deltas.
        * If signs disagree, overwrite with the new contribution.
        * If signs agree, update using a weighted average based on the temperature.

3. **Final Merge**: Add the accumulated deltas to the base model parameters.

This methodology ensures that:
- Higher priority models have more influence on the final result.
- Sign conflicts are resolved in favor of higher priority models.
- The temperature parameter allows control over how much influence each model has in areas of agreement.

## Advanced Features

### Memory Efficiency

- Uses chunked processing to handle large tensors
- Implements memory-efficient versions of key operations (e.g., `memory_efficient_magprune`)
- Utilizes CUDA streams for better GPU memory management

### Customizable Pruning

The `calculate_p` function determines the drop rate for each variant model, allowing for fine-grained control over the pruning process.

### SafeTensors Support

The script uses the SafeTensors format for efficient and safe tensor storage and loading.

## Limitations and Considerations

- Only works with models that share the same architecture (homologous models)
- Performance may vary depending on the similarity and quality of the variant models
- Requires significant computational resources for large language models
- The effectiveness of the merge depends on the chosen priority order of variant models
- The dynamic drop rate calculation may need tuning for optimal results with different model sets

## Contributing

Contributions to Circuit-Merge! are welcome. Please submit issues and pull requests on the project's GitHub repository.

## License

Circuit-Merge is released under the MIT License. See the LICENSE file for details.
