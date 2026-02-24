import os
import json
import random

data_root = "../data/strawberry/complete"
splits_dir = "../deepsdf/experiments/splits"

os.makedirs(splits_dir, exist_ok=True)

all_files = [f for f in os.listdir(data_root) if f.endswith('.ply')]

# Strip .ply extension for deepsdf instance names
all_instances = sorted([f[:-4] for f in all_files])

random.seed(42)
random.shuffle(all_instances)

split_idx = int(len(all_instances) * 0.8)
train_instances = sorted(all_instances[:split_idx])
test_instances = sorted(all_instances[split_idx:])

train_json = {"StrawberryDataset": {"Strawberry": train_instances}}
test_json = {"StrawberryDataset": {"Strawberry": test_instances}}

with open(os.path.join(splits_dir, "strawberry_train.json"), "w") as f:
    json.dump(train_json, f, indent=2)

with open(os.path.join(splits_dir, "strawberry_test.json"), "w") as f:
    json.dump(test_json, f, indent=2)

print(f"Generated {len(train_instances)} training paths and {len(test_instances)} testing paths.")
print(f"Splits saved to {splits_dir}")