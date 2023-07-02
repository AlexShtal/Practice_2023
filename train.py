import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

"""
df = pd.read_csv("Energy_Consumption_Dataset/powerconsumption.csv")

df['Datetime'] = pd.to_datetime(df['Datetime'], format='%m/%d/%Y %H:%M')

df.insert(loc=0,column="Minute",value=df["Datetime"].dt.minute)
df.insert(loc=0,column="Hour",value=df["Datetime"].dt.hour)
df.insert(loc=0,column="Day",value=df["Datetime"].dt.day)
df.insert(loc=0,column="Month",value=df["Datetime"].dt.month)

df = df.drop(columns = "Datetime", axis=1)

model = Sequential()
model.add(LSTM(9,return_sequences = 1, input_shape = (1, 9)))
model.add(Dense(18, activation = 'relu'))
model.add(Dense(9, activation = 'relu'))
model.add(Dense(3, activation = "linear"))

model.compile(loss="mean_squared_error", optimizer="adam")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

X_train = train_df.drop(columns = ["PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3"], axis=1).values.reshape(-1, 1, 9)
y_train = train_df[["PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3"]].values.reshape(-1, 3)

X_test = test_df.drop(columns = ["PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3"], axis=1).values.reshape(-1, 1, 9)
y_test = test_df[["PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3"]].values.reshape(-1, 3)

model.fit(X_train, y_train, epochs=50, verbose=100)

preds = model.predict(X_test, verbose=0)
mape = np.mean(np.abs((y_test - preds) / np.maximum(y_test, 1))) * 100
print(mape.round(2))

model.save("Model", save_format="h5")
"""

df = pd.read_csv("Energy_Consumption_Dataset/powerconsumption.csv")

df['Datetime'] = pd.to_datetime(df['Datetime'], format='%m/%d/%Y %H:%M')

df.insert(loc=0,column="Minute",value=df["Datetime"].dt.minute)
df.insert(loc=0,column="Hour",value=df["Datetime"].dt.hour)
df.insert(loc=0,column="Day",value=df["Datetime"].dt.day)
df.insert(loc=0,column="Month",value=df["Datetime"].dt.month)

df = df.drop(columns = "Datetime", axis=1)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

X_train = train_df.drop(columns = ["PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3"], axis=1).values.reshape(-1, 9)
y_train = train_df[["PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3"]].values.reshape(-1, 3)

X_test = test_df.drop(columns = ["PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3"], axis=1).values.reshape(-1, 9)
y_test = test_df[["PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3"]].values.reshape(-1, 3)

dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, enable_categorical=True)

params = {"objective": "reg:squarederror", "tree_method": "hist"}

n = 1000
model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n
)

preds = model.predict(dtest_reg)

mapedf = np.mean(np.abs((y_test - preds) / y_test)) * 100
mape = np.mean(np.abs((y_test - preds) / np.maximum(y_test, 1))) * 100  
mae = mean_absolute_error(y_test, preds)
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

# Print the model performance metrics
print("Metrics of model performance:")
print("__________________________________________________________________")
print(f"Model Percentage Mean Absolute Error: {mape.round(2)}%")
print(f"Mean Absolute Error: {mae.round(2)}")
print(f"Mean Squared Error: {mse.round(2)}")
print(f"Root Mean Squared Error: {rmse.round(2)}")
print(f"R^2: {r2.round(2)}")
print(f"Percentage Mean Absolute Error: {mapedf.round(2)}%")
print("__________________________________________________________________")


model.save_model("model_xgb.json")