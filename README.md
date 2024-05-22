import yfinance as yf
adani=yf.download("ADANIENT.NS", start='2016-01-01', end='2019-12-31',interval='1d')
data=adani.copy()
data[ : ]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cufflinks as cf
import math
cf.set_config_file(offline=True)
# Here I am plotting graph of the stock price over the time frame which I have choosen
data.iplot(theme = "solar" , title="CANDLE STICK PATTERN OF ADANI ENT.",xTitle="Date", yTitle="price",kind="candle")
SMA CROSSOVERS
# Here I have done Calculation for SMA
data_sma = data[['Close']].copy()
short_sma_values = []
short_window_size = 50

for i in range(len(data) - short_window_size + 1):
    sma = sum(data_sma['Close'][i:i + short_window_size]) / short_window_size
    short_sma_values.append(sma)

data_sma['SHORT SMA'] = [None] * (short_window_size - 1) + short_sma_values

data_sma
long_sma_values = []
long_window_size = 200

for i in range(len(data) - long_window_size + 1):
    sma = sum(data_sma['Close'][i:i + long_window_size]) / long_window_size
    long_sma_values.append(sma)

data_sma['LONG SMA'] = [None] * (long_window_size - 1) + long_sma_values
data_sma
# Here I am plotting graphs for SMA of window size 50 AND 200
data_sma.iplot(xTitle='Date', yTitle='Price', title='Stock Price with Simple Moving Average (SMA CROSSOVERS)',
              width=2, colors=['blue', 'red', 'green'], theme='solar')
position_sma=np.where(data_sma['SHORT SMA']>data_sma['LONG SMA'],1,-1 )
data_sma['POSITION']=position_sma
data_sma
####  A DEATH CROSS occurs when the 50-day SMA crosses below the 200-day SMA sign of a strong bearish trend 
#### A GOLDEN CROSS occurs when the 50-day SMA crosses above the 200-day SMA sign of a strong bullish trend 
EMA CROSSOVERS¶
# Here I have done Calculation for EMA
data_ema = data[['Close']].copy()
ema_valuesshort = [data_ema.iloc[0, 0]]  

period = short_window_size 
multiplier = 2 / (period + 1)
for i in range(1, len(data_ema)):
    ema_today = (data_ema.iloc[i, 0] - ema_valuesshort[i - 1]) * multiplier + ema_valuesshort[i - 1]
    ema_valuesshort.append(ema_today)

data_ema['Short EMA values '] = ema_valuesshort


ema_valueslong = [data_ema.iloc[0, 0]]  

period = long_window_size 
multiplier = 2 / (period + 1)
for i in range(1, len(data_ema)):
    ema_today = (data_ema.iloc[i, 0] - ema_valueslong[i - 1]) * multiplier + ema_valueslong[i - 1]
    ema_valueslong.append(ema_today)

data_ema['Long EMA values '] = ema_valueslong
print(data_ema)
# Here I am plotting graphs for EMA
data_ema.iplot(title='Stock Price with Exponential Moving Average (EMA)', xTitle='Date', yTitle='Price',color=['blue','red','green'],theme='solar')
MACD
# Here I have done calculation for MACD
data_macd = data[['Close']].copy()
# choosing the standard value for the window size
short_window = 12  # Short-term EMA window
long_window = 26 # Long-term EMA window
signal_window = 9  # Signal line window

short_ema = [data_macd.iloc[0, 0]]  
period = short_window
multiplier = 2 / (period + 1)
for i in range(1, len(data_macd)):
    ema_today = (data_macd.iloc[i, 0] - short_ema[i - 1]) * multiplier + short_ema[i - 1]
    short_ema.append(ema_today)
short_ema=np.array(short_ema)


long_ema = [data_macd.iloc[0, 0]]  
period = long_window 
multiplier = 2 / (period + 1)
for i in range(1, len(data_macd)):
    ema_today = (data_macd.iloc[i, 0] - long_ema[i - 1]) * multiplier + long_ema[i - 1]
    long_ema.append(ema_today)
long_ema=np.array(long_ema)    
macd_line = short_ema - long_ema
macd_line=np.array(macd_line)
data_macd['MACD LINE'] = macd_line


signal_line = [data_macd.iloc[0, 1]]
period = signal_window 
multiplier = 2 / (period + 1)
for i in range(1, len(data_ema)):
    ema_today = (data_macd.iloc[i, 1] - macd_line[i - 1]) * multiplier + macd_line[i - 1]
    signal_line.append(ema_today)
signal_line=np.array(signal_line)

data_macd['SIGNAL LINE'] = signal_line

histogram = macd_line - signal_line

data_macd['HISTOGRAM'] = histogram
position_macd=np.where(data_macd['MACD LINE']>data_macd['SIGNAL LINE'],1,-1)
data_macd['position macd']=position_macd
data_macd
dataframe_macd=data_macd[['MACD LINE','SIGNAL LINE','HISTOGRAM']] 
# Here I am plotting graphs for EMA
dataframe_macd.iplot(xTitle='Date', yTitle='MACD', title='MACD Indicator', width=2, colors=['blue', 'red', 'green'],
             mode='lines', theme='solar')
BOLLINGER BANDS¶
data_bb=data[['Close']].copy()
# Here I am calculating bollinger band  
window= 20
middle_band=data_bb[['Close']].rolling(window).mean()
standard_deviation=data_bb[['Close']].rolling(window).std()
upper_band=middle_band-2*(standard_deviation)
lower_band=middle_band+2*(standard_deviation)
distance=data_bb-middle_band
data_bb.dropna(inplace=True) #dropping the intial rows which don't have values
position=np.where(data_bb<lower_band,1,np.nan)
#oversold condition :buy
position=np.where(data_bb>upper_band,-1,position)
# overbrought condition :sell
position=np.where(distance*distance.shift(1)<0,0,position)
# crossing middle band :hold
position= pd.DataFrame(position)
position=position.ffill().fillna(0)
position=np.array(position)
data_bb['MIDDLE BAND'] = middle_band
data_bb['UPPER BAND'] = upper_band
data_bb['LOWER BAND'] = lower_band
# Here I am plotting graphs for Bollinger Bands

data_bb.iplot(xTitle='Date', yTitle='Price', title='Bollinger Bands', width=2, colors=['blue', 'white', 'red', 'green'],
               mode='lines', theme='solar')
data_bb['DISTANCE'] = distance
data_bb['POSITION'] = position
data_bb
STOCHASTIC OSCILLATOR
data_so=data[['Close']].copy()
period =14 # choosing standard value 
roll_low=data['Low'].rolling(period).min()
roll_high=data['High'].rolling(period).max()
k=(data['Close']-data['Low'])/(data['High']-data['Low'])*(100)
moving_avg=3 # standard value
d=k.rolling(moving_avg).mean()
data_so['roll low']=roll_low
data_so['roll high' ]= roll_high
data_so['k']=k
data_so['d']=d
data_so.iplot(xTitle='Date', yTitle='Price', title='STOCHASTIC OSCILLATOR', width=2, colors=['blue', 'white','red', 'green','pink'],
               mode='lines', theme='solar')
position_so=np.where(data_so['k']>data_so['d'],1,-1)
data_so['position so']=position_so
data_so
COMBINING STOCHASTIC OSCILLATOR AND MACD
def combine_indicators(data_so, data_macd):
    signals = []
    position = 0  
    
    for i in range(len(data_so)):
        so_position = data_so['position so'][i]
        macd_position = data_macd['position macd'][i]
        
        if so_position == 1 and macd_position == 1: 
            position = 1
            signals.append('1')
        elif so_position == -1 and macd_position == -1: 
            position = -1
            signals.append('-1')
        else:  # hold signal
            signals.append('0')
    
    return signals
signals = combine_indicators(data_so, data_macd)
data_so['signals'] = signals
data_so['signals']
this combination is telling that should i buy sell or hold
data_frame=data[['Close']].copy()
data_frame['LSMA']=data_sma['LONG SMA'].copy()
data_frame['SSMA']=data_sma['SHORT SMA'].copy()
data_frame['LEMA']=data_ema['Long EMA values '].copy()
data_frame['SEMA']=data_ema['Short EMA values '].copy()
data_frame['MACDLINE']=data_macd['MACD LINE'].copy()
data_frame['SIGNALLINE']=data_macd['SIGNAL LINE'].copy()
data_frame['K']=data_so['k'].copy()
data_frame['D']=data_so['d'].copy()
data_frame['BBMIDDLE']=data_bb['MIDDLE BAND'].copy()
data_frame['BBUPPER']=data_bb['UPPER BAND'].copy()
data_frame['BBLOWER']=data_bb['LOWER BAND'].copy()
data_frame
data_frame['Signal'] = 'Hold'  
buy_condition = (data_frame['SSMA'] > data_frame['LSMA']) & \
                (data_frame['SEMA'] > data_frame['LEMA']) & \
                (data_frame['MACDLINE'] > data_frame['SIGNALLINE']) & \
                (data_frame['K'] > data_frame['D'])

sell_condition = (data_frame['SSMA'] < data_frame['LSMA']) & \
                 (data_frame['SEMA'] < data_frame['LEMA']) & \
                 (data_frame['MACDLINE'] < data_frame['SIGNALLINE']) & \
                 (data_frame['K'] < data_frame['D'])

data_frame.loc[buy_condition, 'Signal'] = 'Buy'
data_frame.loc[sell_condition, 'Signal'] = 'Sell'
data_frame['Signal']
data_frame = data_frame.dropna()
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
X = data_frame.drop(columns=['Signal']).values
y = data_frame['Signal'].values
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
# One-hot encode target labels
label_encoder = LabelEncoder()
one_hot_encoder = OneHotEncoder(categories='auto')
y_encoded = label_encoder.fit_transform(y)
y_encoded = one_hot_encoder.fit_transform(y_encoded.reshape(-1, 1)).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)
np.random.seed(1)
W = np.random.randn(X_train.shape[1], y_train.shape[1])
b = np.zeros((1, y_train.shape[1]))
def forward_propagation(X, W, b):
    return softmax(np.dot(X, W) + b)
learning_rate = 0.01
epochs = 1000
for epoch in range(epochs):
    y_pred = forward_propagation(X_train, W, b)
    grad = y_pred - y_train
    W -= learning_rate * np.dot(X_train.T, grad)
    b -= learning_rate * np.sum(grad, axis=0, keepdims=True)
y_pred_test = forward_propagation(X_test, W, b)
accuracy = np.mean(np.argmax(y_pred_test, axis=1) == np.argmax(y_test, axis=1)) * 100
print(f"Accuracy: {accuracy}%")
from sklearn.metrics import roc_auc_score, f1_score
auc_roc_score = roc_auc_score(y_test, y_pred_test, multi_class='ovr')*100
f1 = f1_score(np.argmax(y_test, axis=1), np.argmax(y_pred_test, axis=1), average='weighted')*100
print(f"AUC-ROC Score: {auc_roc_score}%")
print(f"F1 Score: {f1}%")
