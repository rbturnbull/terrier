import pandas as pd
from typing import List
import random
from functools import partial
from pathlib import Path
from torch import nn
from fastai.data.core import DataLoaders
import torchapp as ta
from fastai.learner import load_learner
from rich.console import Console
from fastai.data.block import DataBlock, TransformBlock, CategoryBlock
from fastai.data.transforms import RandomSplitter
console = Console()
from hierarchicalsoftmax import HierarchicalSoftmaxLoss, SoftmaxNode
from hierarchicalsoftmax import metrics, inference
import csv
import torch
from .famdb import FamDB
from corgi.tensor import dna_seq_to_tensor
from corgi.models import ConvClassifier
from fastcore.foundation import mask2idxs
from fastai.data.transforms import IndexSplitter
from rich.progress import track
from fastai.metrics import accuracy


def greedy_attribute_accuracy(prediction_tensor, target_tensor, root, attribute):
    prediction_nodes = inference.greedy_predictions(prediction_tensor=prediction_tensor, root=root)
    prediction_attributes = torch.as_tensor( [getattr(node, attribute) for node in prediction_nodes], dtype=int).to(target_tensor.device)

    target_nodes = [root.node_list[target] for target in target_tensor]
    target_attributes = torch.as_tensor( [getattr(node, attribute) for node in target_nodes], dtype=int).to(target_tensor.device)

    return (prediction_attributes == target_attributes).float().mean()


class DictionaryGetter:
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, key):
        value = self.dictionary[key]
        return value


class SetSplitter:
    def __init__(self, data):
        self.data = set(data)

    def __call__(self, objects):
        validation_indexes = mask2idxs(object in self.data for object in objects)
        return IndexSplitter(validation_indexes)(objects)


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



class Terrier(FamDBObject, ta.TorchApp):
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
        training:Path = ta.Param(default=None, help="The path to a list of accessions to use for training."),
        validation:Path = ta.Param(default=None, help="The path to a list of accessions to use as validation."),
        repeatmasker_only:bool = False,
    ) -> DataLoaders:
        """
        Creates a FastAI DataLoaders object which Terrier uses in training and prediction.

        Args:
            inputs (Path): The input file.
            batch_size (int, optional): The number of elements to use in a batch for training and prediction. Defaults to 32.

        Returns:
            DataLoaders: The DataLoaders object.
        """
        print("Reading Dfam")
        self.famdb = FamDB(famdb)
        print("Done")
        self.accession_to_node_id = {}
        self.accession_to_repeatmasker_type = {}

        accessions = set()


        ########hack
        # with open("category.csv", "w") as csv:
        #     print("accession", "curated", "node_id", "length", file=csv, sep=",", flush=True)

        #     print('open')
        #     validation_accessions = set(Path(validation).read_text().strip().split("\n"))
        #     print('validation_accessions', len(validation_accessions))
        #     for accession in self.famdb.get_family_accessions():
        #         print(accession)
        #         if accession in accessions:
        #             continue
        #         family = self.famdb.get_family_by_accession(accession)
        #         if not hasattr(family, "classification") or not family.classification:
        #             continue

        #         classification = family.classification.replace("root;","")
        #         assert classification
        #         assert classification in self.classification_nodes
        #         node = self.classification_nodes[classification]
        #         node_id = self.classification_tree.node_to_id[node]
        #         # self.accession_to_node_id[accession] = node_id
        #         self.accession_to_repeatmasker_type[accession] = node.repeat_masker_type
        #         print(accession, node_id, accession in validation_accessions, node_id, family.length, file=csv, sep=",", flush=True)
        # assert False

        # with open("repeatmasker_type.csv", "w") as csv:
        #     print("accession", "repeat_masker_type", "validation", "node_id", "length", file=csv, sep=",", flush=True)

        #     print('open')
        #     validation_accessions = set(Path(validation).read_text().strip().split("\n"))
        #     print('validation_accessions', len(validation_accessions))
        #     for accession in self.famdb.get_family_accessions():
        #         print(accession)
        #         if accession in accessions:
        #             continue
        #         family = self.famdb.get_family_by_accession(accession)
        #         if not hasattr(family, "classification") or not family.classification:
        #             continue
        #         # if family.length < 64:
        #         #     continue
        #         classification = family.classification.replace("root;","")
        #         assert classification
        #         assert classification in self.classification_nodes
        #         node = self.classification_nodes[classification]
        #         node_id = self.classification_tree.node_to_id[node]
        #         # self.accession_to_node_id[accession] = node_id
        #         self.accession_to_repeatmasker_type[accession] = node.repeat_masker_type
        #         print(accession, node.repeat_masker_type, accession in validation_accessions, node_id, family.length, file=csv, sep=",", flush=True)
        # assert False
        ########hack
        
        print("Getting list of accessions")
        if training:
            accessions_to_try = Path(training).read_text().strip().split("\n")
        else:
            accessions_to_try = list(self.famdb.get_family_accessions())

        if validation:
            if max_count:
                # if we are limiting the number of training items, then shuffle the list to make it a random sample
                random.seed(42)
                random.shuffle(accessions_to_try)
            validation_accessions = set(Path(validation).read_text().strip().split("\n"))
            accessions_to_try = list(validation_accessions) + accessions_to_try

        for accession in track(accessions_to_try, "Looking up accessions:"):
            if accession in accessions:
                continue
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
            self.accession_to_repeatmasker_type[accession] = node.repeat_masker_type
            
            accessions.add(accession)
            if max_count and len(accessions) >= max_count:
                print(f"Stopping at {max_count} accessions")
                break

        splitter = SetSplitter(validation_accessions) if validation else RandomSplitter()

        get_x = SequenceGetter(famdb=self.famdb)
        blocks = [TransformBlock, CategoryBlock if repeatmasker_only else TransformBlock]
        get_y = DictionaryGetter(self.accession_to_repeatmasker_type if repeatmasker_only else self.accession_to_node_id)
        self.repeatmasker_only = repeatmasker_only
        datablock = DataBlock(
            blocks=blocks,
            get_x=get_x,
            get_y=get_y,
            splitter=splitter,
        )

        return datablock.dataloaders(accessions, bs=batch_size)

    def loss_func(self):
        if self.repeatmasker_only:
            return nn.CrossEntropyLoss()
        return HierarchicalSoftmaxLoss(root=self.classification_tree)

    def metrics(self):
        if self.repeatmasker_only:
            return [accuracy]

        return [
            partial(metrics.greedy_accuracy, root=self.classification_tree), 
            partial(metrics.greedy_f1_score, root=self.classification_tree),
            partial(metrics.greedy_accuracy_depth_one, root=self.classification_tree), 
            partial(metrics.greedy_accuracy_depth_two, root=self.classification_tree), 
            partial(greedy_attribute_accuracy, root=self.classification_tree, attribute="repeat_masker_type_id"), 
        ]

    def model(
        self,
        corgi:Path = ta.Param(
            default=None,
            help="A pretrained corgi exported app.",
        ),
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
        output_size = len(self.repeatmasker_type_dict.keys()) if self.repeatmasker_only else self.classification_tree.layer_size
        if corgi:
            corgi_learner = load_learner(corgi)
            final_in_features = list(corgi_learner.model.final.modules())[1].in_features

            
            corgi_learner.model.final = nn.Sequential(
                nn.Linear(in_features=final_in_features, out_features=final_in_features, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=final_in_features, out_features=output_size, bias=final_bias),
            )
            return corgi_learner.model

        return ConvClassifier(
            num_embeddings=5,  # i.e. the size of the vocab which is N, A, C, G, T
            kernel_size=kernel_size,
            factor=factor,
            cnn_layers=cnn_layers,
            num_classes=output_size,
            kernel_size_maxpool=kernel_size_maxpool,
            final_bias=final_bias,
            dropout=dropout,
            cnn_dims_start=cnn_dims_start,
        )

    def monitor(self):
        self.repeatmasker_only = True # HACK
        if self.repeatmasker_only:
            return "accuracy"
        return "greedy_f1_score"

    def inference_dataloader(
        self,
        learner,
        file: List[Path] = ta.Param(None, help="A fasta file with sequences to be classified."),
        max_seqs: int = None,
        batch_size:int = 1,
        **kwargs,
    ):
        self.masked_dataloader = dataloaders.MaskedDataloader(files=file, device=learner.dls.device, batch_size=batch_size, min_length=64, max_seqs=max_seqs)
        self.categories = learner.dls.vocab
        return self.masked_dataloader

    def output_results(
        self,
        results,
        csv: Path = ta.Param(default=None, help="A path to output the results as a CSV."),
        **kwargs,
    ):
        repeat_details = pd.DataFrame(self.repeat_details, columns=["file", "accession", "start", "end"])
        predictions_df = pd.DataFrame(results[0].numpy(), columns=self.categories)
        results_df = pd.concat(
            [repeat_details, predictions_df],
            axis=1,
        )
        results_df['prediction'] = results_df[self.categories].idxmax(axis=1)
        if csv:
            console.print(f"Writing results for {len(results_df)} repeats to: {csv}")
            results_df.to_csv(csv, index=False)
        else:
            print("No output file given.")

        return results_df