from tensorflow import keras as ks
from matplotlib import pyplot as plt
import datetime as dt
import pandas

path = "Practice_2023\Energy_Consumption_Dataset\powerconsumption.csv"
ds = pandas.read_csv(path)

df['Datetime'] = pd.to_datetime(df['Datetime'], format='%m/%d/%Y %H:%M')

df = df.set_index(df["Datetime"])
df = df.drop("Datetime", axis = 1)
df.insert(loc=0,column="Month",value=ds.index.month)
