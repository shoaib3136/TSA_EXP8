# Ex.No: 08     MOVINTG AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 
### Developed By: Shaik Shoaib Nawaz
### Register No: 212222240094


### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv("/content/KwhConsumptionBlower78_1.csv")
print("Shape of the dataset:", data.shape)
print("First 50 rows of the dataset:")
print(data.head(50))

# Plot the first 50 values of the "Consumption" column
plt.plot(data['Consumption'].head(50))
plt.title('First 50 values of the "Consumption" column')
plt.xlabel('Index')
plt.ylabel('Power Consumption')
plt.show()

# 1. Moving Average (MA) Model
# MA model with a lag order of 13
lag_order = 13
ma_model = ARIMA(data['Consumption'], order=(0, 0, lag_order))  # (p=0, d=0, q=lag_order)
ma_model_fit = ma_model.fit()

# Predictions from MA model
ma_predictions = ma_model_fit.predict(start=lag_order, end=len(data)-1)
ma_mse = mean_squared_error(data['Consumption'][lag_order:], ma_predictions)
print('MA Model Mean Squared Error (MSE):', ma_mse)

# Plot the original data vs MA model predictions
plt.plot(data['Consumption'][lag_order:], label='Original Data')
plt.plot(ma_predictions, label='MA Predictions')
plt.title('MA Model Predictions vs Original Data')
plt.xlabel('Index')
plt.ylabel('Power Consumption')
plt.legend()
plt.show()

# 2. Exponential Smoothing Model
# Exponential smoothing with trend and seasonal components
exp_smoothing_model = ExponentialSmoothing(data['Consumption'], trend='add', seasonal=None, seasonal_periods=None)
exp_smoothing_fit = exp_smoothing_model.fit()

# Predictions from Exponential Smoothing
exp_smoothing_predictions = exp_smoothing_fit.fittedvalues
exp_smoothing_mse = mean_squared_error(data['Consumption'], exp_smoothing_predictions)
print('Exponential Smoothing Mean Squared Error (MSE):', exp_smoothing_mse)

# Plot the original data vs Exponential Smoothing predictions
plt.plot(data['Consumption'], label='Original Data')
plt.plot(exp_smoothing_predictions, label='Exponential Smoothing Predictions')
plt.title('Exponential Smoothing Predictions vs Original Data')
plt.xlabel('Index')
plt.ylabel('Power Consumption')
plt.legend()
plt.show()

```

### OUTPUT:

#### original data and fitted value :
![alt text](image-7.png)

#### Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF) :
![alt text](image-8.png)


#### Plot the original data and predictions :
![alt text](image-9.png)




### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
