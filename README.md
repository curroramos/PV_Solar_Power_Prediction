# PV_solar_power_prediction

'''
python train.py -m CNN_LSTM -w 168 -e 30
'''


python train.py -m Simpler_RNN -w 168 -e 30 \
& python train.py -m Simple_RNN -w 168 -e 30 \
& python train.py -m LSTM -w 168 -e 30 \
& python train.py -m LSTM_stacked -w 168 -e 30 \
& python train.py -m Bidirectional_LSTM -w 168 -e 30 \
& python train.py -m CNN_LSTM -w 168 -e 30
