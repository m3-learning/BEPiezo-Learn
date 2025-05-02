from abc import ABC, abstractmethod

class BaseFitter(ABC):
    
    def __init__(self, dataset: Dataset):
        self._dataset = dataset
    
    def __getattr__(self, name):
        # Delegate attribute access
        return getattr(self._dataset, name)
    
    def fit_all(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    @abstractmethod
    def fit(self):
        raise NotImplementedError("Subclasses must implement this method")
    