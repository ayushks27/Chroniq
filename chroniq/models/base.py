from abc import ABC, abstractmethod

class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models in the pipeline.
    """
    
    @abstractmethod
    def fit(self, train_data, val_data=None):
        """
        Fits the model to training data, optionally using validation data.
        """
        pass
        
    @abstractmethod
    def predict(self, test_data):
        """
        Generates predictions for the target period.
        """
        pass
