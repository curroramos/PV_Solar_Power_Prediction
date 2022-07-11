# PV_solar_power_prediction

Final Degree Project.
Degree in Industrial Engineering. University of Seville.

This repository hosts the files of the project *Deep Learning Applications to Optimise an EV Charging Station* related with the PV generation forecasting.

## Project breakdown
- Data processing: Public data processing and preparing in order to feed the time series models.
- Training and benchmarking: Train different neural network models on the gathered data and evaluate and compare their performance on the value prediction task.
- Deployment: Build test functions to make inferences and predictions on new data

## Content
The content is arranged in different folders:
- `Root folder`: contains basic [train.py](train.py) script, also contains a [train_demo.ipynb](train_demo.ipynb) and [test_demo.ipynb](test_demo.ipynb) jupyter notebooks to get started into training models and making inferences.
- `data_preprocessing\`: contains files related to first data processing and visualization. Also contains the datasets in [raw_data](raw_data), [preprocessed_data](preprocessed_data), [processed_data](processed_data) 

```bash
.
├── data_processing
│   ├── data_processing.ipynb
│   ├── data_python.py
│   └── processed_data
├── LICENSE
├── Matlab_data
│   ├── azimuthPreference.m
│   ├── checkSizes.m
│   ├── EarthEphemeris.m
│   ├── readMetFile.m
│   ├── readMetFile_modified.m
│   ├── solarRad.mat
│   ├── sunang.m
│   └── sunslope.m
├── model_utils.py
├── plot_utils.py
├── README.md
├── Results
│   ├── train_Bidirectional_LSTM_ws168_epochs30_results
│   ├── train_Bidirectional_LSTM_ws24_epochs30_results
        ...
│   ├── train_Simpler_RNN_ws96_epochs30_results
│   └── train_Simpler_RNN_ws96_epochs3_results
├── test_demo.ipynb
└── train.py
```

## Get started
Please refer to installation

## Usage

To perform a train use this command:

```sh
python train.py -m $MODEL -w $WINDOW_SIZE -e $EPOCHS
```

Examples:

```
python train.py -m Simpler_RNN -w 168 -e 30 \
python train.py -m Simple_RNN -w 168 -e 30 \
python train.py -m LSTM -w 168 -e 30 \
python train.py -m LSTM_stacked -w 168 -e 30 \
python train.py -m Bidirectional_LSTM -w 168 -e 30 \
python train.py -m CNN_LSTM -w 168 -e 30
```

## Main outcomes
- Model benchmarking
    - Main results for each model in [Results]() directory

## Results
### Prediction benchmarking
![alt text]()

![alt text]()

### Real time showcase
![alt text]()
