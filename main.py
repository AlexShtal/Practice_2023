from tensorflow import keras as ks
from matplotlib import pyplot as plt
import datetime as dt
import pandas

path = "Practice_2023\Energy_Consumption_Dataset\powerconsumption.csv"
ds = pandas.read_csv(path)

print(ds.shape)
print(ds)

# ds['Datetime'] = pandas.to_datetime(ds['Datetime'], format='%b/%d/ %I:%M %p').dt.strftime('%b %d %H:%M:%S.%f')
# ds['Datetime'] = ds['Datetime'].apply(lambda x: dt.datetime.strptime(x,'%b %d %H:%M:%S.%f') if type(x)==str else pd.NaT)

