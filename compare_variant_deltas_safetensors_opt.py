import sys
import argparse
from pathlib import Path
import numpy as np
import torch
from safetensors import safe_open
from typing import Dict, List, Tuple
from dataclasses import dataclass
import os
import multiprocessing as mp
from tqdm import tqdm

EPSILON = 1e-5 # Small threshold for float comparison

@dataclass
class ComparisonResults:
    total_params: int = 0
    changed_params1: int = 0
    changed_params2: int = 0
    changed_params_both: int = 0
    positive_changes1: int = 0
    negative_changes1: int = 0
    positive_changes2: int = 0
    negative_changes2: int = 0
    overlap_positive_changes1: int = 0
    overlap_negative_changes1: int = 0
    overlap_positive_changes2: int = 0
    overlap_negative_changes2: int = 0
    max_diff1: float = 0
    max_diff2: float = 0
    sum_diff1: float = 0
    sum_diff2: float = 0
    sum_magnitude1: float = 0
    sum_magnitude2: float = 0
    sum_positive_magnitude1: float = 0
    sum_negative_magnitude1: float = 0
    sum_positive_magnitude2: float = 0
    sum_negative_magnitude2: float = 0
    sum_overlap_magnitude1: float = 0
    sum_overlap_magnitude2: float = 0
    sum_same_direction: int = 0
    sum_overlap_diff: float = 0
    max_overlap_diff: float = 0

    def update(self, other):
        for field in self.__annotations__:
            if field != 'avg_diff1' and field != 'avg_diff2':
                setattr(self, field, getattr(self, field) + getattr(other, field))

    def calculate_final_metrics(self):
        self.avg_diff1 = self.sum_diff1 / self.total_params if self.total_params > 0 else 0
        self.avg_diff2 = self.sum_diff2 / self.total_params if self.total_params > 0 else 0
        self.pct_changed1 = (self.changed_params1 / self.total_params) * 100 if self.total_params > 0 else 0
        self.pct_changed2 = (self.changed_params2 / self.total_params) * 100 if self.total_params > 0 else 0
        self.pct_changed_both = (self.changed_params_both / self.total_params) * 100 if self.total_params > 0 else 0

        self.avg_magnitude1 = self.sum_magnitude1 / self.changed_params1 if self.changed_params1 > 0 else 0
        self.avg_magnitude2 = self.sum_magnitude2 / self.changed_params2 if self.changed_params2 > 0 else 0
        self.avg_positive_magnitude1 = self.sum_positive_magnitude1 / self.positive_changes1 if self.positive_changes1 > 0 else 0
        self.avg_negative_magnitude1 = self.sum_negative_magnitude1 / self.negative_changes1 if self.negative_changes1 > 0 else 0
        self.avg_positive_magnitude2 = self.sum_positive_magnitude2 / self.positive_changes2 if self.positive_changes2 > 0 else 0
        self.avg_negative_magnitude2 = self.sum_negative_magnitude2 / self.negative_changes2 if self.negative_changes2 > 0 else 0
        self.avg_overlap_magnitude1 = self.sum_overlap_magnitude1 / self.changed_params_both if self.changed_params_both > 0 else 0
        self.avg_overlap_magnitude2 = self.sum_overlap_magnitude2 / self.changed_params_both if self.changed_params_both > 0 else 0

        if self.changed_params_both > 0:
            self.directional_agreement = (self.sum_same_direction / self.changed_params_both) * 100
            self.avg_overlap_diff = self.sum_overlap_diff / self.changed_params_both
        else:
            self.directional_agreement = 0
            self.avg_overlap_diff = 0

def load_tensors_from_directory(directory: str) -> Dict[str, str]:
    all_tensors = {}
    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.safetensors'):
            file_path = os.path.join(directory, filename)
            with safe_open(file_path, framework="pt", device="cpu") as f:
                all_tensors.update({k: file_path for k in f.keys()})
    return all_tensors

def process_tensor(args):
    tensor_name, base_file, variant1_file, variant2_file = args
    results = ComparisonResults()

    with safe_open(base_file, framework="pt", device="cpu") as base_f, \
         safe_open(variant1_file, framework="pt", device="cpu") as variant1_f, \
         safe_open(variant2_file, framework="pt", device="cpu") as variant2_f:
        
        base_tensor = base_f.get_tensor(tensor_name)
        variant1_tensor = variant1_f.get_tensor(tensor_name)
        variant2_tensor = variant2_f.get_tensor(tensor_name)

        if base_tensor.shape != variant1_tensor.shape or base_tensor.shape != variant2_tensor.shape:
            return results, (tensor_name, "Shape mismatch")

        if base_tensor.dtype != variant1_tensor.dtype or base_tensor.dtype != variant2_tensor.dtype:
            print("Type mismatch! downconvert to fp16")
            #return results, (tensor_name, "Type mismatch")
            variant1_tensor = variant1_f.get_tensor(tensor_name).half()
            variant2_tensor = variant2_f.get_tensor(tensor_name).half()

        try:
            delta1 = (variant1_tensor - base_tensor).cpu().numpy()
            delta2 = (variant2_tensor - base_tensor).cpu().numpy()

            # Apply EPSILON threshold to zero out small deltas
            delta1[np.abs(delta1) < EPSILON] = 0
            delta2[np.abs(delta2) < EPSILON] = 0

            results.total_params += base_tensor.numel()
            results.changed_params1 += np.count_nonzero(delta1)
            results.changed_params2 += np.count_nonzero(delta2)

            results.positive_changes1 += np.sum(delta1 > 0)
            results.negative_changes1 += np.sum(delta1 < 0)
            results.positive_changes2 += np.sum(delta2 > 0)
            results.negative_changes2 += np.sum(delta2 < 0)

            overlap_mask = (delta1 != 0) & (delta2 != 0)
            results.changed_params_both += np.sum(overlap_mask)

            results.overlap_positive_changes1 += np.sum((delta1 > 0) & overlap_mask)
            results.overlap_negative_changes1 += np.sum((delta1 < 0) & overlap_mask)
            results.overlap_positive_changes2 += np.sum((delta2 > 0) & overlap_mask)
            results.overlap_negative_changes2 += np.sum((delta2 < 0) & overlap_mask)

            same_direction = ((delta1 > 0) & (delta2 > 0)) | ((delta1 < 0) & (delta2 < 0))
            results.sum_same_direction += np.sum(same_direction & overlap_mask)

            results.sum_magnitude1 += np.sum(np.abs(delta1))
            results.sum_magnitude2 += np.sum(np.abs(delta2))
            results.sum_positive_magnitude1 += np.sum(np.abs(delta1[delta1 > 0]))
            results.sum_negative_magnitude1 += np.sum(np.abs(delta1[delta1 < 0]))
            results.sum_positive_magnitude2 += np.sum(np.abs(delta2[delta2 > 0]))
            results.sum_negative_magnitude2 += np.sum(np.abs(delta2[delta2 < 0]))
            results.sum_overlap_magnitude1 += np.sum(np.abs(delta1[overlap_mask]))
            results.sum_overlap_magnitude2 += np.sum(np.abs(delta2[overlap_mask]))

            overlap_diff = np.abs(delta1[overlap_mask] - delta2[overlap_mask])
            max_diff = np.max(overlap_diff) if overlap_diff.size > 0 else 0

            results.sum_overlap_diff += np.sum(overlap_diff)
            results.max_overlap_diff = max(results.max_overlap_diff, max_diff)

            return results, (tensor_name, max_diff, base_tensor.shape, base_tensor.dtype)

        except Exception as e:
            return results, (tensor_name, f"Error: {str(e)}")

def compare_safetensors_files(base_dir: str, variant1_dir: str, variant2_dir: str):
    print(f"Base directory: {base_dir}")
    print(f"Variant 1 directory: {variant1_dir}")
    print(f"Variant 2 directory: {variant2_dir}")
    print()

    print("Loading tensor metadata...")
    base_tensors = load_tensors_from_directory(base_dir)
    variant1_tensors = load_tensors_from_directory(variant1_dir)
    variant2_tensors = load_tensors_from_directory(variant2_dir)

    common_tensors = set(base_tensors.keys()).intersection(set(variant1_tensors.keys()), set(variant2_tensors.keys()))
    only_in_base = set(base_tensors.keys()) - set(variant1_tensors.keys()).union(set(variant2_tensors.keys()))
    only_in_variant1 = set(variant1_tensors.keys()) - set(base_tensors.keys()).union(set(variant2_tensors.keys()))
    only_in_variant2 = set(variant2_tensors.keys()) - set(base_tensors.keys()).union(set(variant1_tensors.keys()))

    print(f"Tensors only in base: {len(only_in_base)}")
    print(f"Tensors only in variant 1: {len(only_in_variant1)}")
    print(f"Tensors only in variant 2: {len(only_in_variant2)}")
    print(f"Common tensors: {len(common_tensors)}")

    print("Comparing tensors:")
    args_list = [(name, base_tensors[name], variant1_tensors[name], variant2_tensors[name]) for name in common_tensors]

    with mp.Pool() as pool:
        results = []
        large_discrepancies = []
        total_results = ComparisonResults()

        for result, discrepancy in tqdm(pool.imap_unordered(process_tensor, args_list), total=len(args_list)):
            results.append(result)
            total_results.update(result)
            if isinstance(discrepancy[1], (int, float)):
                large_discrepancies.append(discrepancy)

    large_discrepancies.sort(key=lambda x: x[1], reverse=True)
    print("\nTop 10 largest discrepancies:")
    for name, diff, shape, dtype in large_discrepancies[:10]:
        print(f"Tensor: {name}, Max Diff: {diff}, Shape: {shape}, Dtype: {dtype}")

    total_results.calculate_final_metrics()
    print_results(total_results)

def print_results(results: ComparisonResults):
    print(f"Total parameters: {results.total_params}")
    print(f"\nVariant 1 Changes:")
    print(f"  Changed parameters: {results.changed_params1} ({results.changed_params1/results.total_params*100:.2f}%)")
    print(f"  Positive changes: {results.positive_changes1} ({results.positive_changes1/results.changed_params1*100:.2f}%)")
    print(f"  Negative changes: {results.negative_changes1} ({results.negative_changes1/results.changed_params1*100:.2f}%)")
    print(f"  Average magnitude of changes: {results.avg_magnitude1:.6f}")
    print(f"  Average magnitude of positive changes: {results.avg_positive_magnitude1:.6f}")
    print(f"  Average magnitude of negative changes: {results.avg_negative_magnitude1:.6f}")

    print(f"\nVariant 2 Changes:")
    print(f"  Changed parameters: {results.changed_params2} ({results.changed_params2/results.total_params*100:.2f}%)")
    print(f"  Positive changes: {results.positive_changes2} ({results.positive_changes2/results.changed_params2*100:.2f}%)")
    print(f"  Negative changes: {results.negative_changes2} ({results.negative_changes2/results.changed_params2*100:.2f}%)")
    print(f"  Average magnitude of changes: {results.avg_magnitude2:.6f}")
    print(f"  Average magnitude of positive changes: {results.avg_positive_magnitude2:.6f}")
    print(f"  Average magnitude of negative changes: {results.avg_negative_magnitude2:.6f}")

    print(f"\nOverlapping Changes:")
    print(f"  Parameters changed in both variants: {results.changed_params_both} ({results.changed_params_both/results.total_params*100:.2f}%)")
    print(f"  Overlapping positive changes in variant 1: {results.overlap_positive_changes1} ({results.overlap_positive_changes1/results.changed_params_both*100:.2f}%)")
    print(f"  Overlapping negative changes in variant 1: {results.overlap_negative_changes1} ({results.overlap_negative_changes1/results.changed_params_both*100:.2f}%)")
    print(f"  Overlapping positive changes in variant 2: {results.overlap_positive_changes2} ({results.overlap_positive_changes2/results.changed_params_both*100:.2f}%)")
    print(f"  Overlapping negative changes in variant 2: {results.overlap_negative_changes2} ({results.overlap_negative_changes2/results.changed_params_both*100:.2f}%)")
    print(f"  Average magnitude of overlapping changes in variant 1: {results.avg_overlap_magnitude1:.6f}")
    print(f"  Average magnitude of overlapping changes in variant 2: {results.avg_overlap_magnitude2:.6f}")

    print(f"\nDirectional Agreement:")
    print(f"  Percentage of overlapping changes in the same direction: {results.directional_agreement:.2f}%")

    print(f"\nOverlap Difference Metrics:")
    print(f"  Average difference in overlapping changes: {results.avg_overlap_diff:.6f}")
    print(f"  Maximum difference in overlapping changes: {results.max_overlap_diff:.6f}")

def main():
    parser = argparse.ArgumentParser(description="Compare a base SafeTensors model with two variant models")
    parser.add_argument("base_model", type=str, help="Base SafeTensors model directory")
    parser.add_argument("variant1", type=str, help="First variant SafeTensors model directory")
    parser.add_argument("variant2", type=str, help="Second variant SafeTensors model directory")
    args = parser.parse_args()

    compare_safetensors_files(args.base_model, args.variant1, args.variant2)

if __name__ == "__main__":
    main()
