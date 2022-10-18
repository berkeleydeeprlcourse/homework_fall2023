import numpy as np
from typing import Any


Prediction = Any


class BaseModel(object):
    def update(self, ob_no, next_ob_no, re_n, terminal_n) -> dict:
        raise NotImplementedError

    def get_prediction(self, ob_no, ac_na, data_statistics) -> Prediction:
        raise NotImplementedError

    def convert_prediction_to_numpy(self, pred: Prediction) -> np.ndarray:
        """Allow caller to be pytorch-agnostic."""
        raise NotImplementedError
