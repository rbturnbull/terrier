from pathlib import Path
from torch import nn
from fastai.data.core import DataLoaders
import torchapp as fa
from rich.console import Console
console = Console()

class Corkie(fa.TorchApp):
    """
    Classifier of Repeats
    """
    def dataloaders(
        self,
        inputs:Path = fa.Param(help="The input file."), 
        batch_size:int = fa.Param(default=32, help="The batch size."),
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which Corkie uses in training and prediction.

        Args:
            inputs (Path): The input file.
            batch_size (int, optional): The number of elements to use in a batch for training and prediction. Defaults to 32.

        Returns:
            DataLoaders: The DataLoaders object.
        """
        raise NotImplemented("Dataloaders function not implemented yet.") 

    def model(
        self,
    ) -> nn.Module:
        """
        Creates a deep learning model for the Corkie to use.

        Returns:
            nn.Module: The created model.
        """
        raise NotImplemented("Model function not implemented yet.") 
        return nn.Sequential(
        )
