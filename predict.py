from tensorflow import keras as ks
import pandas as pd
import xgboost as xgb

"""
path = "DataBase\Model_Input"
ds = pandas.read_csv(path)

model = ks.models.load_model("Model")

input = ds.values.reshape(-1, 1, 9)

output = model.predict(input).T

ds.insert(loc = 9, column = "Power_Consumption_Zone1", value=output[0][0]) 
ds.insert(loc = 10, column = "Power_Consumption_Zone2", value=output[1][0]) 
ds.insert(loc = 11, column = "Power_Consumption_Zone3", value=output[2][0]) 

ds.to_csv("DataBase/Model_Output.csv", index=False)
"""
model = xgb.Booster()
model.load_model("model_xgb.json")

path = "DataBase\Model_Input"
input = pd.read_csv(path)

dinput = xgb.DMatrix(input.values.reshape(-1, 9))

preds = model.predict(dinput)

preds = pd.DataFrame(preds)
preds = preds.set_axis(["Power_Consumption_Zone1",
                        "Power_Consumption_Zone2",
                        "Power_Consumption_Zone3"], axis=1)

output = pd.concat([input, preds], axis=1)
print(output)

output.to_csv("DataBase\Model_Output", index=False)