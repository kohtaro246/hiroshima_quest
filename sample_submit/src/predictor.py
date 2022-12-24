import pandas as pd
from tutorial_transformer import make_prediction
import numpy as np
import torch


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.

        Returns:
            bool: The return value. True for success.
        """
        cls.model = "../model/best.pt"

        return True


    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input: Data of the sample you want to make inference from (dict)

        Returns:
            list: Inference for the given input.

        """
        def _missing_value_process(df):
            df['value'] = df['value'].replace({'M':0.0, '*':0.0, '-':0.0, '--': 0.0, '**':0.0})
            df['value'] = df['value'].fillna(0.0)
            df['value'] = df['value'].astype(float)
            return df

        stations = input['stations']
        waterlevel = input['waterlevel']
        rainfall = input['rainfall']
        #tidelevel = input['tidelevel']
        waterlevel = pd.merge(pd.DataFrame(stations, columns=['station']), pd.DataFrame(waterlevel))
        waterlevel = _missing_value_process(waterlevel)
        rainfall = pd.DataFrame(rainfall)
        rainfall = rainfall[rainfall["station"]=="栗谷"]
        rainfall = _missing_value_process(rainfall)
        input_data = pd.merge(waterlevel, rainfall, on="hour")
        hours = input_data["hour"].tolist()
        input_data_value = input_data[["value_x", "value_y"]].to_numpy()
        assert input_data_value.shape[0] % 24 == 0
        input_data_value = np.stack([input_data_value[24*i:24*(i+1)] for i in range(input_data_value.shape[0]//24)])
        device = "cpu"
        
        pred = make_prediction(input_data_value, cls.model, device)

        input_data["value"] = pred
        input_data = input_data.rename(columns={"station_x": "station"})
        prediction = input_data[["hour", "station", "value"]].to_dict('records')
        return prediction
