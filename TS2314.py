#By Lorenzo Ng in March 2020. Call or whatsapp me on 51122647

import numpy as np 
import pandas as pd 
from pandas.plotting import register_matplotlib_converters
import matplotlib.pylab as plt 
from matplotlib.pylab import rcParams
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
register_matplotlib_converters()

#Read File
df = pd.read_csv('2314.csv', parse_dates=['Date'])
df.head()
print('There are ' + str(df.AdjClose.size) + ' datas.')
series = df.AdjClose.values
print(df.AdjClose.describe())
print(type(series))
print(series.size)

#Plot Adj Closing Graph
x_axis = df.Date
y_axis = df.AdjClose
plt.plot(x_axis, y_axis)
plt.xlabel('Date')
plt.ylabel('Adjusted Closing')
plt.title('2314.HK')
plt.show()

#Plot Adj Closing Mean Graph
closingmean = df.AdjClose.rolling(window = 5).mean()
x_axis = df.Date
y_axis = closingmean
plt.plot(x_axis, y_axis)
plt.xlabel('Date')
plt.ylabel('Adjusted Closing Mean')
plt.title('2314.HK')
plt.show()

#Creating Series
adjc = pd.DataFrame(series)
series_df = pd.concat([adjc,adjc.shift(1)],axis=1)
series_df.columns = ['Closing','ForecastClosing']
print(series_df.tail())

testseries = series_df[1:]
series_error = mean_squared_error(testseries.Closing,testseries.ForecastClosing)
print(np.sqrt(series_error))

#ARIMA 
#ACF
plot_acf(series)
plt.show()
#PACF
plot_pacf(series)
plt.show()

series_train = series[0:210]
series_test = series[210:247]

#Training
seriestrain = ARIMA(series_train, order= (2,0,3))
series_fit = seriestrain.fit()
print(series_fit.aic)
forecastclosing = series_fit.forecast(steps = 37)[0]
print(forecastclosing)

#Errors
print('The error is listed below:')
print(np.sqrt(mean_squared_error(series_test,forecastclosing)))
