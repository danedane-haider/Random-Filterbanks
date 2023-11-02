import pickle
import numpy as np
import os

# Load results
results_dir = "/Users/felixperfler/Downloads/models"

results = []
for file in os.listdir(results_dir):
    if file.endswith(".pkl"):
        with open(os.path.join(results_dir, file), "rb") as f:
            results.append(pickle.load(f))

criterium = 'kappa_val_kappa'
best_model_params = None
lowest_criterium = np.inf
lowest_criterium_epoch = None

for result in results:
    if np.min(result[criterium]) < lowest_criterium:
        lowest_criterium = np.min(result[criterium])
        best_model_params = result['parameter_config']
        lowest_criterium_epoch = np.argmin(result[criterium])

print(f"Best model parameters: {best_model_params} at epoch {lowest_criterium_epoch} with {criterium} {lowest_criterium}")