from functools import partial
from pathlib import Path
from torch import nn
from fastai.data.core import DataLoaders
import torchapp as fa
from rich.console import Console
console = Console()
from hierarchicalsoftmax import HierarchicalSoftmaxLoss, SoftmaxNode
from hierarchicalsoftmax import metrics, inference
import csv
import torch


def greedy_attribute_accuracy(prediction_tensor, target_tensor, root, attribute):
    prediction_nodes = inference.greedy_predictions(prediction_tensor=prediction_tensor, root=root)
    prediction_attributes = torch.as_tensor( [getattr(node, attribute) for node in prediction_nodes], dtype=int).to(target_tensor.device)

    target_nodes = [root.node_list[target] for target in target_tensor]
    target_attributes = torch.as_tensor( [getattr(node, attribute) for node in target_nodes], dtype=int).to(target_tensor.device)

    return (prediction_attributes == target_attributes).float().mean()


class Corkie(fa.TorchApp):
    """
    Classifier of Repeats
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classification_tree = SoftmaxNode(name="root")
        self.classification_nodes = {}
        self.repeatmasker_types = []
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
                
                repeat_masker_type = row['repeatmasker_type']
                repeat_masker_type_id = None
                if repeat_masker_type:
                    if not repeat_masker_type in self.repeatmasker_types:
                        self.repeatmasker_types.append(repeat_masker_type)
                    repeat_masker_type_id = self.repeatmasker_types.index(repeat_masker_type)

                node = SoftmaxNode(
                    name=lineage[-1], 
                    parent=parent, 
                    title=row['title'], 
                    repeat_masker_type=repeat_masker_type,
                    repeat_masker_type_id=repeat_masker_type_id,
                )
                self.classification_nodes[lineage_string] = node

        self.repeatmasker_type_dict = {key: value for key, value in enumerate(self.repeatmasker_types)}

        try:
            text_file = Path("classification_tree.txt")
            text_file.write_text(str(self.classification_tree.render()))
            self.classification_tree.render(filepath="classification_tree.svg")
            self.classification_tree.render(filepath="classification_tree.png")
        except Exception as err:
            print(f"Cannot render classification tree {err}")

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
            partial(greedy_attribute_accuracy, root=self.classification_tree, attribute="repeat_masker_type_id"), 
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