import pandas as pd


class ScoringService(object):
    @classmethod
    def get_model(cls, model_path):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.

        Returns:
            bool: The return value. True for success.
        """
        cls.model = None

        return True


    @classmethod
    def predict(cls, input):
        """Predict method

        Args:
            input: Data of the sample you want to make inference from (dict)

        Returns:
            list: Inference for the given input.

        """
        stations = input['stations']
        waterlevel = input['waterlevel']
        merged = pd.merge(pd.DataFrame(stations, columns=['station']), pd.DataFrame(waterlevel))
        merged['value'] = merged['value'].replace({'M':0.0, '*':0.0, '-':0.0, '--': 0.0, '**':0.0})
        merged['value'] = merged['value'].fillna(0.0)
        merged['value'] = merged['value'].astype(float)
        prediction = merged[['hour', 'station', 'value']].to_dict('records')

        return prediction
