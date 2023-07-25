import re
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
from hierarchicalsoftmax import metrics, inference, greedy_predictions
import csv
import torch
from corgi.tensor import dna_seq_to_tensor
from corgi.models import ConvClassifier
from corgi.dataloaders import SeqIODataloader
from fastcore.foundation import mask2idxs
from fastai.data.transforms import IndexSplitter
from Bio import SeqIO
from rich.progress import track
from fastai.metrics import accuracy
import xarray as xr
from torchapp.apps import call_func
from .models import VectorOutput

from .loss import FocalLoss
from .famdb import FamDB
from .dataloaders import MaskedDataloader

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

    def render_classification_tree(self):
        try:
            text_file = Path("classification_tree.txt")
            text_file.write_text(str(self.classification_tree.render()))
            self.classification_tree.render(filepath="classification_tree.svg")
            self.classification_tree.render(filepath="classification_tree.png")
        except Exception as err:
            print(f"Cannot render classification tree {err}")

    def setup_repbase_classification_tree(self, label_smoothing=0.0, gamma=None):
        self.classification_tree = SoftmaxNode(name="root", label_smoothing=label_smoothing, gamma=gamma)
        self.classification_nodes = {}
        
        prev = self.classification_tree
        parent = self.classification_tree
        indentation = 0
        with open(Path(__file__).parent/"data/repbase.tree.txt") as f:
            for line in f:
                my_indent = line.count("\t")
                if my_indent > indentation:
                    assert my_indent == indentation + 1
                    parent = prev
                    indentation = my_indent
                elif my_indent < indentation:
                    while my_indent < indentation:
                        parent = parent.parent
                        indentation -= 1
                name = line.strip()
                prev = SoftmaxNode(name, parent=parent, count=0, label_smoothing=label_smoothing, gamma=gamma)
                self.classification_nodes[name] = prev
        self.classification_tree.set_indexes()

    def setup_dfam_classification_tree(self):
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

    def dataloaders(
        self,
        repbase:Path = ta.Param(None, help="RepBase in FASTA format."), 
        famdb:Path = ta.Param(help="The FamDB file (hdf5) input file from Dfam."), 
        batch_size:int = ta.Param(default=1, help="The batch size."),
        max_count:int = ta.Param(default=None, help="The maximum number of families to train with. Default unlimited."),
        training:Path = ta.Param(default=None, help="The path to a list of accessions to use for training."),
        validation:Path = ta.Param(default=None, help="The path to a list of accessions to use as validation."),
        label_smoothing:float = ta.Param(default=0.0, min=0.0, max=1.0, help="The amount of label smoothing to use."),
        gamma:float = ta.Param(default=None, min=0.0, help="The gamma value for using Focal Loss."),
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
        if repbase:
            print("Reading RepBase")
            from .repbase import RepBase, RepBaseGetter
            self.repbase = RepBase(repbase)
            self.setup_repbase_classification_tree(label_smoothing=label_smoothing, gamma=gamma)
            get_x = RepBaseGetter(self.repbase)
        else:
            self.repbase = None
            self.setup_dfam_classification_tree()
            print("Reading Dfam")
            self.famdb = FamDB(famdb)
            get_x = SequenceGetter(self.famdb)
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

        for accession in accessions_to_try:
        # for accession in track(accessions_to_try, "Looking up accessions:"):
            if accession in accessions:
                continue

            if self.repbase:
                record = self.repbase[accession]
                classification = record.description.split("\t")[1]
                if classification not in self.classification_nodes:
                    print("classification not in list")
                    assert False
                node = self.classification_nodes[classification]
                node_id = self.classification_tree.node_to_id[node]
                self.accession_to_node_id[accession] = node_id
            else:
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

        blocks = [TransformBlock, CategoryBlock if repeatmasker_only else TransformBlock]
        get_y = DictionaryGetter(self.accession_to_repeatmasker_type if repeatmasker_only else self.accession_to_node_id)
        self.repeatmasker_only = repeatmasker_only
        datablock = DataBlock(
            blocks=blocks,
            get_x=get_x,
            get_y=get_y,
            splitter=splitter,
        )

        dls = datablock.dataloaders(accessions, bs=batch_size)
        if self.repeatmasker_only:
            self.vocab = dls.vocab
        return dls

    def loss_func(self,gamma:float=0.0):
        if self.repeatmasker_only:
            if gamma == 0.0:
                return nn.CrossEntropyLoss()
            else:
                return FocalLoss(gamma=gamma)
        return HierarchicalSoftmaxLoss(root=self.classification_tree)

    def metrics(self):
        if self.repeatmasker_only:
            return [accuracy]

        return [
            partial(metrics.greedy_accuracy, root=self.classification_tree), 
            partial(metrics.greedy_f1_score, root=self.classification_tree),
            partial(metrics.greedy_accuracy_depth_one, root=self.classification_tree), 
            partial(metrics.greedy_accuracy_depth_two, root=self.classification_tree), 
            # partial(greedy_attribute_accuracy, root=self.classification_tree, attribute="repeat_masker_type_id"), 
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
        output_size = len(self.vocab) if self.repeatmasker_only else self.classification_tree.layer_size
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
        # self.repeatmasker_only = True # HACK
        # if self.repeatmasker_only:
        #     return "accuracy"
        return "greedy_f1_score"

    def inference_dataloader(
        self,
        learner,
        fasta: List[Path] = ta.Param(None, help="A fasta file with sequences to be classified."),
        max_seqs: int = None,
        max_length:int=25_000,
        batch_size:int = 1,
        **kwargs,
    ):        
        self.dataloader = SeqIODataloader(
            files=fasta, 
            device=learner.dls.device, 
            batch_size=batch_size, 
            min_length=1,
            max_length=max_length,
            max_seqs=max_seqs,
            format="fasta",
        )
        # self.masked_dataloader = MaskedDataloader(files=fasta, format="fasta", device=learner.dls.device, batch_size=batch_size, min_length=128, max_seqs=max_seqs, max_repeats=max_repeats)
        # only works for repbase
        self.setup_repbase_classification_tree()
        return self.dataloader

    def output_results(
        self,
        results,
        # repeat_masker_out:Path = ta.Param(default=None, help="The .out file from repeat masker."),
        # repeat_masker_replace:Path = ta.Param(default=None, help="A path to write a replacement repeat masker out file."),
        output_csv: Path = ta.Param(default=None, help="A path to output the results as a CSV."),
        output_fasta: Path = ta.Param(default=None, help="A path to output the results in FASTA format."),
        image_dir: Path = ta.Param(default=None, help="A directory to output the results as images."),
        image_format:str = "svg",
        image_threshold:float = 0.005,
        **kwargs,
    ):
        if self.vector:
            classification_results = results[0][0]
        else:
            classification_results = results[0]
        
        classification_probabilities = inference.node_probabilities(results[0], root=self.classification_tree)
        # greedy_predictions = inference.greedy_predictions(results[0], root=self.classification_tree)

        chunk_details = pd.DataFrame(self.dataloader.chunk_details, columns=["file", "original_id", "chunk"])
        category_names = [str(node) for node in self.classification_tree.node_list if not node.is_root]
        leaf_names = [str(node) for node in self.classification_tree.leaves]
        predictions_df = pd.DataFrame(classification_probabilities.numpy(), columns=category_names)
        results_df = pd.concat(
            [chunk_details.drop(columns=['chunk']), predictions_df],
            axis=1,
        )

        # Average over chunks
        results_df["order"] = results_df.index
        results_df = results_df.groupby(["file", "original_id"]).mean().reset_index()

        # sort to get original order
        results_df = results_df.sort_values(by="order").drop(columns=["order"])
        
        # results_df['max_leaf_probability_prediction'] = results_df[leaf_names].idxmax(axis=1)

        # Get new tensors now that we've averaged over chunks
        classification_probabilities = torch.as_tensor(results_df[category_names].to_numpy()) 
        # get greedy predictions which can use the raw activation or the softmax probabilities
        greedy_predictions = inference.greedy_predictions(classification_probabilities, root=self.classification_tree)

        results_df['greedy_prediction'] = [str(node) for node in greedy_predictions]

        results_df['accession'] = results_df['original_id'].apply(lambda x: x.split("#")[0])
        def get_original_classification(original_id:str):
            if "#" in original_id:
                return original_id.split("#")[1]
            return "null"
        
        def get_prediction_probability(row):
            prediction = row["greedy_prediction"]
            return row[prediction]
        
        results_df['probability'] = results_df.apply(get_prediction_probability, axis=1)
        results_df['original_classification'] = results_df['original_id'].apply(get_original_classification)

        # Reorder columns
        results_df = results_df[["file", "accession", "greedy_prediction", "probability", "original_id", "original_classification" ] + category_names]

        # Output images
        if image_dir:
            console.print(f"Writing inference probability renders to: {image_dir}")
            image_dir = Path(image_dir)
            image_paths = []
            for _, row in results_df.iterrows():
                filepath = row['file']
                accession = row['accession']
                image_path = image_dir / Path(filepath).name / f"{accession}.{image_format}"
                image_paths.append(image_path)
            inference.render_probabilities(
                root=self.classification_tree, 
                filepaths=image_paths,
                probabilities=classification_probabilities,
                predictions=greedy_predictions,
                threshold=image_threshold,
            )

        if output_fasta:
            console.print(f"Writing results for {len(results_df)} repeats to: {output_fasta}")
            with open(output_fasta, "w") as fasta_out:
                for file in self.dataloader.files:
                    for record in SeqIO.parse(file, "fasta"):
                        original_id = record.id
                        row = results_df.loc[results_df.original_id == original_id]
                        if len(row) == 0:
                            SeqIO.write(record, fasta_out, "fasta")
                            continue

                        accession = row['accession'].item()
                        original_classification = row["original_classification"].item()
                        prediction = row["greedy_prediction"].item()
                        
                        new_id = f"{accession}#{prediction}"
                        record.id = new_id
                        
                        # Adapt description
                        record.description = record.description.replace(original_id, "")
                        last_bracket = record.description.rfind(")")
                        if last_bracket == -1:
                            record.description = f"{record.description} ( "
                        else:
                            record.description = record.description[:last_bracket].rstrip() + ", "

                        new_probability = row[prediction].values[0]
                        record.description = f"{record.description} original classification = {original_classification}, classification probability = {new_probability:.2f} )"

                        SeqIO.write(record, fasta_out, "fasta")


        # repeat_details = pd.DataFrame(self.dataloader.repeat_details, columns=["file", "accession", "start", "end"])
        # prediction_nodes = inference.greedy_predictions(classification_results, root=self.classification_tree)
        # repeat_details["category"] = prediction_nodes
        # repeat_details["hierarchical_classification"] = [" > ".join([str(anc) for anc in node.ancestors[1:] + (str(node),) ]) for node in prediction_nodes]

        # current_node = 0
        # with open(repeat_masker_out, "r") as f, open(repeat_masker_replace, "w") as out:
        #     for line_number, line in enumerate(f):
        #         if line_number >= 3: # after header
        #             m = re.match(r"^\s*(\S+\s+){11}", line)
        #             if m is None:
        #                 breakpoint()
        #             start, stop = m.span(1)
        #             prediction = str(prediction_nodes[current_node])

        #             # error checking 
        #             m = re.match(r"^\s*(\S+\s+){5}(\d+)\s+(\d+)", line)
        #             assert m is not None
        #             seq_start = int(m.group(2))
        #             seq_end = int(m.group(3))
        #             if seq_start != self.masked_dataloader.repeat_details[current_node][2]:
        #                 breakpoint()
        #             # if seq_end != self.masked_dataloader.repeat_details[current_node][3]:
        #             #     breakpoint()

        #             line = line[:start] + prediction + " "*(stop-start-len(prediction)) + line[stop:]
        #             current_node += 1

        #         out.write(line)
        #         if current_node >= len(prediction_nodes):
        #             break

        # classification_probabilities = torch.softmax(classification_results, axis=1)
        # predictions_df = pd.DataFrame(classification_probabilities.numpy(), columns=self.categories)
        # results_df = pd.concat(
        #     [repeat_details, predictions_df],
        #     axis=1,
        # )
        # results_df['prediction'] = results_df[self.categories].idxmax(axis=1)
        # results_df["prediction_exclude_unknown"] = results_df[[c for c in self.categories if c != "Unknown"]].idxmax(axis=1)

        if output_csv:
            console.print(f"Writing results for {len(results_df)} repeats to: {output_csv}")
            results_df.to_csv(output_csv, index=False)

        else:
            print("No output file given.")

        # if self.vector:
        #     # x = results_df.to_xarray()
        #     # breakpoint()
        #     # embeddings = xr.DataArray(results[0][1], dims=("accession", "embedding"))
        #     # embeddings.to_netcdf("embeddings.nc")
        #     torch.save(results[0][1], "embeddings.pkl")

        return results_df

    def __call__(
        self, 
        gpu: bool = ta.Param(True, help="Whether or not to use a GPU for processing if available."), 
        vector: bool = ta.Param(False, help="Whether or not to save the penultimate layer activations as a vector."), 
        **kwargs
    ):
        # Check if CUDA is available
        gpu = gpu and torch.cuda.is_available()

        # Open the exported learner from a pickle file
        path = call_func(self.pretrained_local_path, **kwargs)
        learner = self.learner_obj = load_learner(path, cpu=not gpu)

        # Create a dataloader for inference
        dataloader = call_func(self.inference_dataloader, learner, **kwargs)

        self.vector = vector
        if vector:
            # adapt the model
            learner.model.final = VectorOutput(learner.model.final)

        results = learner.get_preds(dl=dataloader, reorder=False, with_decoded=False, act=self.activation(), cbs=self.inference_callbacks())

        # Output results
        output_results = call_func(self.output_results, results, **kwargs)
        return output_results if output_results is not None else results
