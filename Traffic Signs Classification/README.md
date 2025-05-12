# Traffic Sign Classification

In this project, Convolutional Neural Networks (CNN) provide lighting using traffic signs. The German Traffic Sign Recognition Benchmark (GTSRB) dataset is managed.

## Installation

To register the required subscriptions:

``` bash
pip install -r requirements.txt
```
## Dataset

The [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html) dataset has just been released. To download the dataset:

1. Visit [GTSRB web systems](https://benchmark.ini.rub.de/gtsrb_dataset.html).

2. Download the dataset from the "Download dataset" section.

3. Extract the downloaded dataset to `data`.

## Usage

For model training:

``` bash
python train.py
```

For making predictions on test details:

``` bash
python predict.py --image_path/to/your/image.jpg
```

## Project Structure

- `data/`: Dataset files
- `models/`: Trained model files
- `src/`: Source codes
- `data_loader.py`: Data loading and preprocessing
- `model.py`: CNN model definition
- `train.py`: Model training
- `predict.py`: Prediction
- `notebooks/`: Jupyter notebooks (for data mining and analysis)
