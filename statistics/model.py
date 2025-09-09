import pickle
import pandas as pd

df = pd.DataFrame(
    [['M', '51-55', 7, 'B', "3", 1, 1]],
    columns=[
        'Gender',
        'Age',
        'Occupation',
        'City_Category',
        'Stay_In_Current_City_Years',
        'Marital_Status',
        'Product_Category_1'
    ]
)

with open(r"F:\Data science\statistics\regrassion.pkl", "rb") as f:
    pipeline = pickle.load(f)

print(pipeline.predict(df))
