# S4ECG: Exploring the impact of long-range interactions for arrhythmia prediction

Welcome to the official GitHub repository for the paper "[S4ECG: Exploring the impact of long-range interactions for arrhythmia prediction]". If you consider this repository useful for you research, we would appreciate a citation of our preprint.

## Table of Contents
- [Setup & Environment](#setup--environment)
- [Datasets](#datasets)
- [Preprocessing](#preprocessing)
- [Classification](#classification)
- [Contribution](#contribution)
- [Acknowledgments](#acknowledgments)

## Setup & Environment

1. **Environment**: The environment details and requirements for Linux systems are provided in the `environment.yml` folder. Please ensure to set up the provided environment to avoid compatibility issues.

2. **Additional Package**: You'll need to install the `cauchy` or other related packages for the S4 model. See the official repository for more details:

	https://github.com/HazyResearch/state-spaces/


## Datasets

We utilize four primary datasets for our research:

- `Icentia11k`: https://physionet.org/content/icentia11k-continuous-ecg/1.0/
- `LTAFDB`: https://physionet.org/content/ltafdb/1.0.0/
- `AFDB`: https://physionet.org/content/afdb/1.0.0/
- `MITDB`: https://www.physionet.org/content/mitdb/1.0.0/

Both datasets should be appropriately placed in the main directory or as instructed by specific scripts.

## Preprocessing

To preprocess the raw data:

Run the `preprocess.py` script:
```
python preprocess.py icentia /path/to/your/dataset.zip
```

This will process the raw data from the aforementioned datasets and prepare them for classification.

## Classification

To run the classification:

1. **Configuration**: Modify the config files located at `./conf/data` to specify the directory containing the preprocessed data.

2. **Training the model**:
Set the desired config file using the `--config-name` flag. Example for a time-series-based model on Icentia11k:
```
python main.py --config-name=config_id_icentia.yaml
```

You can use other config settings by specifying different `.yaml` files located in the `./classification/code/conf` directory. We provide config files for the best-performing model architectures for time series and spectrograms as input representations.

For ood test the path of trained model should be denoted as:
```
python main.py --config-name=config_ood_ltafdb.yaml trainer.eval_only=/path/best_model.ckpt
```


## Contribution

Feel free to submit pull requests, raise issues, or suggest enhancements to improve the project. We appreciate community contributions!

## Acknowledgments
This work partly builds on the S4 layer kindly provided by https://github.com/HazyResearch/state-spaces/

---

For any further queries or concerns, please refer to the official publication or contact the project maintainers.
