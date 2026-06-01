from abc import ABC, abstractmethod

class BaseEmbedings(ABC):

    @abstractmethod
    def get_model(self):
        pass