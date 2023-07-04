import pandas as pd
import xgboost as xgb


model = xgb.Booster()
model.load_model("model_xgb.json")

path = "DataBase\Model_Input"
input = pd.read_csv(path)

dinput = xgb.DMatrix(input.values.reshape(-1, 9))

preds = model.predict(dinput)

preds = pd.DataFrame(preds)
preds = preds.set_axis(["PowerConsumption_Zone1",
                        "PowerConsumption_Zone2",
                        "PowerConsumption_Zone3"], axis=1)

output = pd.concat([input, preds], axis=1)
print(output)

output.to_csv("DataBase\Model_Output", index=False)
