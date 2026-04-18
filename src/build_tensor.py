import pandas as pd
import numpy as np
from pathlib import Path

def build_tensor(data_dir="../data", processed_subdir="processed", metadata_file="metadata.csv", n_times=100, n_channels=2, max_samples_per_class=400):
    data_dir = Path(data_dir)
    processed_dir = data_dir / processed_subdir
    metadata_path = data_dir / metadata_file


    # Load metadata
    metadata = pd.read_csv(metadata_path)
    metadata = metadata.set_index('oid')

    # List processed files
    processed_files = list(processed_dir.glob("*_processed.csv"))
    print(f"Processed files: {len(processed_files)}")

    # Collect objects by class for balanced sampling
    objects_by_class = {}

    for file in processed_files:
        df = pd.read_csv(file)
        object_id = file.stem.replace('_detections_processed','').replace('_processed','')
        # Check for both filters and enough points per filter
        if set(df['filter_id']) != {1,2}:
            print(f"Skip {object_id}: missing filters")
            continue
        if df.groupby('filter_id').size().min() < n_times:
            print(f"Skip {object_id}: not enough points per filter")
            continue
        # Sort and reshape: [n_times, n_channels]
        df = df.sort_values(['filter_id','relative_time'])
        arr_g = df[df['filter_id']==1]['mag_pred'].values[:n_times]
        arr_r = df[df['filter_id']==2]['mag_pred'].values[:n_times]
        arr = np.stack([arr_g, arr_r], axis=1) # shape (n_times, n_channels)

        # Get class label
        try:
            class_name = metadata.loc[object_id, 'class_name']
        except KeyError:
            print(f"Skip {object_id}: not in metadata")
            continue

        if class_name not in objects_by_class:
            objects_by_class[class_name] = []
        objects_by_class[class_name].append(arr)

    # Apply elegant sampling
    X_final = []
    y_final = []
    
    print("\n--- Final Dataset Balance ---")
    for class_name, samples in objects_by_class.items():
        n_samples = len(samples)
        if n_samples > max_samples_per_class:
            print(f"Sampling {class_name}: {n_samples} -> {max_samples_per_class}")
            # Randomly shuffle and slice
            np.random.shuffle(samples)
            samples = samples[:max_samples_per_class]
        else:
            print(f"Keeping {class_name}: {n_samples}")
            
        X_final.extend(samples)
        y_final.extend([class_name] * len(samples))

    X = np.stack(X_final) # shape (n_objects, n_times, n_channels)
    y = np.array(y_final)
    print(f"Tensor X shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    # Save for training
    np.save(processed_dir / "X_lightcurves.npy", X)
    pd.Series(y).to_csv(processed_dir / "y_labels.csv", index=False)
    print("Saved X_lightcurves.npy and y_labels.csv")

if __name__ == "__main__":
    build_tensor()