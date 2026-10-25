
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import scanpy as sc

# Read arguments from Omnibenchmark
parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", required=True)
parser.add_argument("--name", required=True)
parser.add_argument("--data.h5ad", dest="data_path", required=True)
parser.add_argument("--data.clusters_truth_num", dest="truth_path", required=True)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

start_time = time.time()

# Load the dataset
data = sc.read_h5ad(args.data_path)
num_cells = data.n_obs

# Get the number of clusters
with open(args.truth_path) as file:
    num_clusters = int(file.read().strip())

# Randomly assign cells to clusters
np.random.seed(args.seed)

leiden = np.random.randint(0, num_clusters, num_cells)
louvain = np.random.randint(0, num_clusters, num_cells)

# Save cluster labels
clusters = pd.DataFrame({
    "cell_id": data.obs_names,
    "leiden": leiden,
    "louvain": louvain
})

clusters.to_csv(
    os.path.join(args.output_dir, args.name + ".clusters.tsv"),
    sep="\t",
    index=False
)

# Create random coordinates for the metrics calculation
pca = pd.DataFrame({
    "cell_id": data.obs_names,
    "PC1": np.random.normal(size=num_cells),
    "PC2": np.random.normal(size=num_cells)
})

pca.to_csv(
    os.path.join(args.output_dir, args.name + ".pca.tsv"),
    sep="\t",
    index=False
)

# Save the running time
run_time = time.time() - start_time

with open(os.path.join(args.output_dir, args.name + ".timings.json"), "w") as file:
    json.dump({"random_clustering": run_time}, file)

# Save clustering information
clustering_info = {
    "resolutions": {"leiden": None, "louvain": None},
    "num_runs": {"leiden": 1, "louvain": 1}
}

with open(os.path.join(args.output_dir, args.name + ".clustering_info.json"), "w") as file:
    json.dump(clustering_info, file)
