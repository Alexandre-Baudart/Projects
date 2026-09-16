from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression

class Calibration(ABC):
    def __init__(self):
        self.calibrator_ = None

    @abstractmethod
    def fit(self, scores, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, scores):
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, scores):
        raise NotImplementedError

    def decision_function(self, scores):
        raise NotImplementedError

class SigmoidCalibration(Calibration):
    """
    Platt scaling using logistic regression.
    """
    def __init__(self):
        super().__init__()

    def fit(self, scores, y):
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)

        self.calibrator_ = LogisticRegression()
        self.calibrator_.fit(scores, y)

        return self

    def predict(self, scores):
        if self.calibrator_ is None:
            raise RuntimeError("The calibrator must be fitted before predict()!")
        
        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)

        return self.calibrator_.predict(scores)

    def predict_proba(self, scores):
        if self.calibrator_ is None:
            raise RuntimeError("The calibrator must be fitted before predict_proba()!")

        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)

        return self.calibrator_.predict_proba(scores)[:, 1]

    def decision_function(self, scores):
        if self.calibrator_ is None:
            raise RuntimeError("The calibrator must be fitted before decision_function()!")

        if scores.ndim == 1:
            scores = scores.reshape(-1, 1)

        return self.calibrator_.decision_function(scores)