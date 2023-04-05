from functools import partial
from pathlib import Path
from torch import nn
from fastai.data.core import DataLoaders
import torchapp as ta
from rich.console import Console
from fastai.data.block import DataBlock, TransformBlock
from fastai.data.transforms import RandomSplitter
console = Console()
from hierarchicalsoftmax import HierarchicalSoftmaxLoss, SoftmaxNode
from hierarchicalsoftmax import metrics
import csv
from .famdb import FamDB
from corgi.tensor import dna_seq_to_tensor
from corgi.models import ConvClassifier


class DictionaryGetter:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, key):
        value = self.dictionary[key]
        return value


class FamDBObject():
    def __init__(self, famdb=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.famdb = famdb

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["famdb"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.famdb = None
    

class SequenceGetter(FamDBObject):
    def __call__(self, accession:str):
        assert self.famdb
        family = self.famdb.get_family_by_accession(accession)
        return dna_seq_to_tensor(family.consensus)


class ClassificationGetter(FamDBObject):
    def __init__(self, famdb, classification_nodes):
        self.famdb = famdb
        self.classification_nodes = classification_nodes

    def __call__(self, accession:str):
        family = self.famdb.get_family_by_accession(accession)
        return dna_seq_to_tensor(family.consensus)



class Corkie(FamDBObject, ta.TorchApp):
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
        self.classification_tree.set_indexes()
        try:
            text_file = Path("classification_tree.txt")
            text_file.write_text(str(self.classification_tree.render()))
            self.classification_tree.render(filepath="classification_tree.svg")
            self.classification_tree.render(filepath="classification_tree.png")
        except Exception as err:
            print(f"Cannot render classification tree {err}")

    def dataloaders(
        self,
        famdb:Path = ta.Param(help="The FamDB file (hdf5) input file from Dfam."), 
        batch_size:int = ta.Param(default=1, help="The batch size."),
        max_count:int = ta.Param(default=None, help="The maximum number of families to train with. Default unlimited."),
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which Corkie uses in training and prediction.

        Args:
            inputs (Path): The input file.
            batch_size (int, optional): The number of elements to use in a batch for training and prediction. Defaults to 32.

        Returns:
            DataLoaders: The DataLoaders object.
        """
        self.famdb = FamDB(famdb)
        self.accession_to_node_id = {}

        accessions = []
        for accession in self.famdb.get_family_accessions():
            family = self.famdb.get_family_by_accession(accession)
            if not hasattr(family, "classification") or not family.classification:
                continue
            if family.length < 64:
                continue
            classification = family.classification.replace("root;","")
            assert classification
            assert classification in self.classification_nodes
            node = self.classification_nodes[classification]
            node_id = self.classification_tree.node_to_id[node]
            self.accession_to_node_id[accession] = node_id
            
            accessions.append(accession)

        if max_count:
            accessions = accessions[:max_count]

        datablock = DataBlock(
            blocks=[TransformBlock, TransformBlock],
            get_x=SequenceGetter(famdb=self.famdb),
            get_y=DictionaryGetter(self.accession_to_node_id),
            splitter=RandomSplitter(),
        )

        return datablock.dataloaders(accessions, bs=batch_size)

    def loss_func(self):
        return HierarchicalSoftmaxLoss(root=self.classification_tree)

    def metrics(self):
        return [
            partial(metrics.greedy_accuracy, root=self.classification_tree), 
            partial(metrics.greedy_f1_score, root=self.classification_tree),
            partial(metrics.greedy_accuracy_depth_one, root=self.classification_tree), 
            partial(metrics.greedy_accuracy_depth_two, root=self.classification_tree), 
        ]

    def model(
        self,
        embedding_dim: int = ta.Param(
            default=8,
            help="The size of the embeddings for the nucleotides (N, A, G, C, T).",
            tune=True,
            tune_min=4,
            tune_max=32,
            log=True,
        ),
        filters: int = ta.Param(
            default=256,
            help="The number of filters in each of the 1D convolution layers. These are concatenated together",
        ),
        cnn_layers: int = ta.Param(
            default=6,
            help="The number of 1D convolution layers.",
            tune=True,
            tune_min=2,
            tune_max=6,
        ),
        kernel_size_maxpool: int = ta.Param(
            default=2,
            help="The size of the pooling before going to the LSTM.",
        ),
        lstm_dims: int = ta.Param(default=256, help="The size of the hidden layers in the LSTM in both directions."),
        final_layer_dims: int = ta.Param(
            default=0, help="The size of a dense layer after the LSTM. If this is zero then this layer isn't used."
        ),
        dropout: float = ta.Param(
            default=0.5,
            help="The amount of dropout to use. (not currently enabled)",
            tune=True,
            tune_min=0.0,
            tune_max=1.0,
        ),
        final_bias: bool = ta.Param(
            default=True,
            help="Whether or not to use bias in the final layer.",
            tune=True,
        ),
        cnn_only: bool = True,
        kernel_size: int = ta.Param(
            default=3, help="The size of the kernels for CNN only classifier.", tune=True, tune_choices=[3, 5, 7, 9]
        ),
        cnn_dims_start: int = ta.Param(
            default=64,
            help="The size of the number of filters in the first CNN layer.",
            tune=True,
            log=True,
            tune_min=32,
            tune_max=1024,
        ),
        factor: float = ta.Param(
            default=2.0,
            help="The factor to multiply the number of filters in the CNN layers each time it is downscaled.",
            tune=True,
            log=True,
            tune_min=0.5,
            tune_max=4.0,
        ),
    ) -> nn.Module:
        """
        Creates a deep learning model for the Corgi to use.
        Returns:
            nn.Module: The created model.
        """
        # TODO - get pretrained
        return ConvClassifier(
            num_embeddings=5,  # i.e. the size of the vocab which is N, A, C, G, T
            kernel_size=kernel_size,
            factor=factor,
            cnn_layers=cnn_layers,
            num_classes=self.classification_tree.layer_size,
            kernel_size_maxpool=kernel_size_maxpool,
            final_bias=final_bias,
            dropout=dropout,
            cnn_dims_start=cnn_dims_start,
        )

    def monitor(self):
        return "greedy_f1_score"

    def output_results(
        self, 
        results, 
        **kwargs
    ):
        print(results)
        return results