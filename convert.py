import pandas as pd

df = pd.read_csv("Energy_Consumption_Dataset\powerconsumption.csv")


df['Datetime'] = pd.to_datetime(df['Datetime'], format='%m/%d/%Y %H:%M')

df.insert(loc=0,column="Month", value=df["Datetime"].dt.month)
df.insert(loc=1,column="Day", value=df["Datetime"].dt.day)
df.insert(loc=2,column="Hour", value=df["Datetime"].dt.hour)
df.insert(loc=3,column="Minute", value=df["Datetime"].dt.minute)
df = df.drop("Datetime", axis=1)

df = df.drop(["PowerConsumption_Zone1", "PowerConsumption_Zone2", "PowerConsumption_Zone3"], axis=1)

df = df[df["Month"] == 12]

df.to_csv("DataBase\Model_Input", index=False)
