from abc import ABC, abstractmethod

# Abstract Factory
class MachineLearningFactory(ABC):
    @abstractmethod
    def create_trainer(self):
        pass

# Concrete Factory de TensorFlow
class TensorFlowFactory(MachineLearningFactory):
    def create_trainer(self):
        return TensorFlowTrainer()

# Concrete Factory de Pytorch
class PytorchFactory(MachineLearningFactory):
    def create_trainer(self):
        return PytorchTrainer()

# Abstract Product
class Trainer(ABC):
    @abstractmethod
    def train(self):
        pass

# Concrete Product de TensorFlow
class TensorFlowTrainer(Trainer):
    def train(self):
        print("Entrenamiento del factory TensorFlow")

# Concrete Product de Pytorch
class PytorchTrainer(Trainer):
    def train(self):
        print("Entrenamiento del factory Pytorch")

# Client
class Client:
    def __init__(self, factory):
        self.factory = factory
        self.trainer = self.factory.create_trainer()

    def execute_training(self):
        self.trainer.train()

# Usage
if __name__ == "__main__":
    tensorflow_factory = TensorFlowFactory()
    pytorch_factory = PytorchFactory()

    tensorflow_client = Client(tensorflow_factory)
    pytorch_client = Client(pytorch_factory)

    tensorflow_client.execute_training()
    pytorch_client.execute_training()