import glob
import pandas as pd 
from pathlib import Path
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# Load astronomical data
data_dir = Path("../data")
processed_dir = data_dir / "processed"
file_list = glob.glob(str(data_dir / "*.csv"))

# Getting all the files in data
for file_path in file_list:
    path_obj = Path(file_path)
    file_name = path_obj.name
    df = pd.read_csv(file_path)
    # if the file is not ok, we ignore it 
    if df.empty: 
        print(f"The file {file_name} is empty.")
        continue

    print(f"Succesfully loaded {file_name} with {len(df)} rows.")
    result_list = []
    for fid in [1,2]: # 1: g-band, 2: r-band
        filter_df = df[df["fid"] == fid].sort_values('mjd')
        # if we don't have enough data we pass
        if len(filter_df) < 5:
            continue

        # 1. Prepare data for this filter
        X = filter_df['mjd'].values.reshape(-1, 1)
        y = filter_df['magpsf'].values
        y_err = filter_df['sigmapsf'].values

        # 2. Scale X to start at 0 (Prevents the convergence warning!)
        X_start = X.min()
        X_scaled = X - X_start

        # 3. Define and Fit GP
        # Note: We use normalize_y=True as we discussed
        kernel = C(1.0) * RBF(length_scale=10.0) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err**2, 
                                    normalize_y=True, n_restarts_optimizer=5)
        gp.fit(X_scaled, y)

        # 4. Create a regular grid of 100 points
        # This is the "Magic Step" for Deep Learning: uniform input size!
        x_grid = np.linspace(X_scaled.min(), X_scaled.max(), 100).reshape(-1, 1)
        y_pred, sigma = gp.predict(x_grid, return_std=True)

        # 5. Store the results in a temporary list or DataFrame
        # We'll save: time_relative, mag_pred, uncertainty, and the filter ID
        for i in range(len(x_grid)):
            result_list.append({
                'object_id': path_obj.stem,
                'filter_id': fid,
                'relative_time': x_grid[i][0],
                'mag_pred': y_pred[i],
                'sigma': sigma[i]
            })
    if result_list: 
        processed_df = pd.DataFrame(result_list)
        output_path = processed_dir / f"{path_obj.stem}_processed.csv"
        processed_df.to_csv(output_path, index = False)
        print(f"   --> Saved: {output_path.name}") # Confirma el guardado
    else:
        print(f"   --> Skip: {path_obj.stem} (Not enough data in filters)")
        

