
from clinical_ts.ecg_utils import *
from clinical_ts.timeseries_utils import *
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

def process_dataset(config):
    """
    config: dict with keys:
        - zip_path: path to source zip file
        - npy_folder: folder to save npy files
        - memmap_folder: folder to save memmap
        - memmap_name: name of memmap file (e.g. 'memmap.npy')
        - recreate_data: bool
        - max_len: int (optional)
    """
    print(f"Processing {config['zip_path']} -> {config['memmap_folder']}")
    df, lbl, mean, std = prepare_icentia(
        config['zip_path'],
        target_folder=config['npy_folder'],
        recreate_data=config.get('recreate_data', False)
    )
    df["label"] = df.data.apply(lambda x: ".".join(str(x).split(".")[:-1]) + "_ann.npy")
    print("saving as memmap...")
    target_folder = Path(config['memmap_folder'])
    source_folder = Path(config['npy_folder'])
    reformat_as_memmap(
        df,
        target_folder / config['memmap_name'],
        data_folder=source_folder,
        delete_npys=False,
        annotation=True,
        max_len=config.get('max_len', 900000000)
    )
    clean_npys(source_folder, 10)

# Example: add as many configs as needed, and select by name
DATASET_CONFIGS = {
    'icentia': {
        'zip_path': '',
        'npy_folder': './datasets/icentia/npy',
        'memmap_folder': './datasets/icentia/memmap',
        'memmap_name': 'memmap.npy',
        'recreate_data': False,
        'max_len': 900000000
    },
    'ltafdb': {
        'zip_path': '',
        'npy_folder': '',
        'memmap_folder': './datasets/icentia/memmap',
        'memmap_name': 'memmap.npy',
        'recreate_data': False,
        'max_len': 900000000
    },
    'icentia': {
        'zip_path': '',
        'npy_folder': './datasets/icentia/npy',
        'memmap_folder': './datasets/icentia/memmap',
        'memmap_name': 'memmap.npy',
        'recreate_data': False,
        'max_len': 900000000
    },
        'icentia': {
        'zip_path': '',
        'npy_folder': './datasets/icentia/npy',
        'memmap_folder': './datasets/icentia/memmap',
        'memmap_name': 'memmap.npy',
        'recreate_data': False,
        'max_len': 900000000
    }
}

# Choose which dataset to preprocess by name
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
        if dataset_name in DATASET_CONFIGS:
            config = DATASET_CONFIGS[dataset_name].copy()
            if len(sys.argv) > 2:
                config['zip_path'] = sys.argv[2]
                print(f"Overriding zip_path with: {config['zip_path']}")
            process_dataset(config)
        else:
            print(f"Unknown dataset name: {dataset_name}\nAvailable: {list(DATASET_CONFIGS.keys())}")
    else:
        print(f"Please provide a dataset name as argument. Available: {list(DATASET_CONFIGS.keys())}")

def clean_npy(filename, clip_value=10):
    x = np.load(filename)
    nans = np.sum(np.isnan(x))
    if nans > 0:
        print(filename, ":", nans, "nans")
        x[np.isnan(x)] = 0
    clips = np.sum(x > clip_value) + np.sum(x < -clip_value)
    if clips > 0:
        print(clips, "clips")
        x = np.clip(x, -clip_value, clip_value)
    if clips > 0 or nans > 0:
        np.save(filename, x)

def clean_npys(folder, clip_value=10):
    for f in tqdm(list(Path(folder).glob("*.npy"))):
        clean_npy(f, clip_value)
