import io
import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import csv
import codecs

from starlette.responses import StreamingResponse

app = FastAPI()


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


def preprocessing(df):

    df.drop(columns=['name', 'selling_price', 'torque'], inplace=True)

    df.loc[df['max_power'] == ' bhp', 'max_power'] = '0 bhp'
    df['mileage'] = df['mileage'].str.strip(' km\/kg| kmpl')
    df['engine'] = df['engine'].str.strip(' CC')
    df['max_power'] = df['max_power'].str.strip(' bhp')

    if len(df) != 1:
        mean_mileage = pd.to_numeric(df['mileage']).mean()
        mean_engine = pd.to_numeric(df['engine']).mean()
        mean_max_power = pd.to_numeric(df['max_power']).mean()
        mean_seats = pd.to_numeric(df['seats']).mean()

        df['mileage'] = df['mileage'].replace("", mean_mileage)
        df['engine'] = df['engine'].replace("", mean_engine)
        df['max_power'] = df['max_power'].replace("", mean_max_power)
        df['seats'] = df['seats'].replace("", mean_seats)

    else:
        df['mileage'] = df['mileage'].replace("", 0)
        df['engine'] = df['engine'].replace("", 0)
        df['max_power'] = df['max_power'].replace("", 0)
        df['seats'] = df['seats'].replace("", 0)

    df = df.astype({'year': 'float', 'km_driven': 'float', 'mileage': 'float', 'engine': 'float', 'max_power': 'float',
                    'seats': 'float'})
    df['seats'] = df['seats'].astype('int')
    df['seats'] = df['seats'].astype('object')

    scaler = joblib.load('scaler.pkl')

    df_real = df.select_dtypes(exclude=['object'])
    df_cat = df.select_dtypes(include=['object'])

    df_real_norm = scaler.transform(df_real)

    if len(df) != 1:
        df_cat = pd.get_dummies(df_cat, drop_first=True)
    else:
        cat_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
        for el in cat_columns:
            temp = df_cat.loc[0, el]
            df_cat.loc[0, el + '_' + str(temp)] = 1
        df_cat.drop(cat_columns, axis=1, inplace=True)

    df_cat['year'] = df_real_norm.T[0]
    df_cat['km_driven'] = df_real_norm.T[1]
    df_cat['mileage'] = df_real_norm.T[2]
    df_cat['engine'] = df_real_norm.T[3]
    df_cat['max_power'] = df_real_norm.T[4]

    columns = joblib.load('train_columns.pkl')

    df_cat = df_cat.reindex(columns=columns, fill_value=0)

    return df_cat


@app.post("/predict_item")
async def predict_item(item: Item):
    item = item.dict()
    car_features = pd.DataFrame.from_dict(item, orient='index').T

    X_cat = preprocessing(car_features)

    model = joblib.load("ridge_model.pkl")
    prediction = model.predict(X_cat.values)

    return prediction[0]


@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):
    data = file.file
    data = csv.reader(codecs.iterdecode(data, 'utf-8'))
    header = data.__next__()
    df = pd.DataFrame(data, columns=header)

    df2 = df.copy(deep=True)

    df_cat = preprocessing(df)

    model = joblib.load("ridge_model.pkl")
    df2['price_prediction'] = model.predict(df_cat.values)
    df2.to_csv('car_with_prediction.csv')

    response = StreamingResponse(io.StringIO(df2.to_csv(index=False)), media_type="text/csv")

    return response
