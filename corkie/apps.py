from functools import partial
from pathlib import Path
from torch import nn
from fastai.data.core import DataLoaders
import torchapp as fa
from rich.console import Console
console = Console()
from hierarchicalsoftmax import HierarchicalSoftmaxLoss, SoftmaxNode
from hierarchicalsoftmax import metrics
import csv


class Corkie(fa.TorchApp):
    """
    Classifier of Repeats
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classification_tree = SoftmaxNode(name="root")
        self.classification_nodes = {}
        with open(Path(__file__).parent/"data/TEClasses.tsv") as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                lineage_string = row['full_name']
                lineage = lineage_string.split(";")
                if len(lineage) == 1:
                    parent = self.classification_tree
                else:
                    parent_lineage_string = ";".join(lineage[:-1])
                    parent = self.classification_nodes[parent_lineage_string]
                
                node = SoftmaxNode(name=lineage[-1], parent=parent, title=row['title'])
                self.classification_nodes[lineage_string] = node

        try:
            self.classification_tree.render(filepath="dfam.svg")
            self.classification_tree.render(filepath="dfam.png")
        except:
            print("Cannot render classification tree")

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

    def loss_func(self):
        return HierarchicalSoftmaxLoss(root=self.classification_tree)

    def metrics(self):
        return [
            partial(metrics.greedy_accuracy, root=self.classification_tree), 
            partial(metrics.greedy_f1_score, root=self.classification_tree),
            partial(metrics.greedy_accuracy_depth_one, root=self.classification_tree), 
            partial(metrics.greedy_accuracy_depth_two, root=self.classification_tree), 
        ]

    def monitor(self):
        return "greedy_accuracy_depth_two"

    def output_results(
        self, 
        results, 
        **kwargs
    ):
        print(results)
        return results