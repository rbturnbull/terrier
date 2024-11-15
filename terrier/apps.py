import pandas as pd
from typing import List
import lightning as L
from functools import partial
from pathlib import Path
from torch import nn
import torchapp as ta
from torchmetrics import Metric
from hierarchicalsoftmax.metrics import RankAccuracyTorchMetric
from hierarchicalsoftmax import HierarchicalSoftmaxLoss, SoftmaxNode
from hierarchicalsoftmax import metrics, inference, greedy_predictions
import torch

# from corgi.models import ConvClassifier
# from corgi.dataloaders import SeqIODataloader
from Bio import SeqIO
from rich.progress import track
from seqbank import SeqBank
from corgi import Corgi
from corgi.seqtree import SeqTree
from polytorch.metrics import HierarchicalGreedyAccuracy

from rich.console import Console
console = Console()

from .repeatmasker import create_repeatmasker_seqtree
from .evaluate import build_confusion_matrix, confusion_matrix_fig, threshold_fig


class Terrier(Corgi):
    """
    Transposable Element Repeat Result classifIER
    """
    @ta.method("super")
    def data(self, **kwargs):
        data = super().data(**kwargs)
        return data

    # def data(
    #     self,
    #     seqtree: Path = ta.Param(help="The seqtree which has the sequences to use."),
    #     seqbank:Path = ta.Param(help="The HDF5 file with the sequences."),
    #     validation_partition:int = ta.Param(default=1, help="The partition to use for validation."),
    #     batch_size: int = ta.Param(default=32, help="The batch size."),
    #     phi:float=ta.Param(default=1.0, tune=True, tune_max=1.2, tune_min=0.8, help="A multiplication factor for the loss at each level of the tree."),
    #     min_length:int = 64,
    #     max_length:int = 4096,
    #     deform_lambda:float = ta.Param(default=None, help="The lambda for the deform transform."),
    #     tips_mode:bool = True,
    # ) -> DataLoaders:
    #     """
    #     Creates a Pytorch Lightning Data Module which Terrier uses in training and validation.

    #     Returns:
    #         DataLoaders: The DataLoaders object.
    #     """
    #     self.validation_partition = validation_partition
    #     dls = super().dataloaders(
    #         seqtree=seqtree,
    #         seqbank=seqbank,
    #         validation_partition=validation_partition,
    #         batch_size=batch_size,
    #         # dataloader_type=dataloader_type,
    #         # deform_lambda=deform_lambda,
    #         tips_mode=tips_mode,
    #         phi=phi,
    #     )
        
    #     before_batch = Pipeline(PadBatch(min_length=min_length, max_length=max_length))
    #     dls.train.before_batch = before_batch
    #     dls.valid.before_batch = before_batch
        
    #     return dls

    @ta.method    
    def metrics(self) -> list[tuple[str,Metric]]:
        rank_accuracy = RankAccuracyTorchMetric(
            root=self.classification_tree, 
            ranks={1:"accuracy_repeatmasker_type", 2:"accuracy_repeatmasker_subtype"},
        )
                
        return [('rank_accuracy', rank_accuracy)]
    
    @ta.method
    def monitor(self):
        return "accuracy_repeatmasker_subtype"

    @ta.method
    def output_results(
        self,
        results,
        output_csv: Path = ta.Param(default=None, help="A path to output the results as a CSV."),
        output_tips_csv: Path = ta.Param(default=None, help="A path to output the results as a CSV which only stores the probabilities at the tips."),
        output_fasta: Path = ta.Param(default=None, help="A path to output the results in FASTA format."),
        image_dir: Path = ta.Param(default=None, help="A directory to output the results as images."),
        image_format:str = "png",
        image_threshold:float = 0.005,
        prediction_threshold:float = ta.Param(default=0.5, help="The threshold value for making hierarchical predictions."),
        **kwargs,
    ):        
        def node_lineage_string(node) -> str:
            if node.is_root:
                return "Unknown"
            return "/".join([str(n) for n in node.ancestors[1:]] + [str(node)])

        classification_probabilities = inference.node_probabilities(results[0], root=self.classification_tree)
        category_names = [node_lineage_string(node) for node in self.classification_tree.node_list if not node.is_root]

        # greedy_predictions = inference.greedy_predictions(results[0], root=self.classification_tree)

        chunk_details = pd.DataFrame(self.dataloader.chunk_details, columns=["file", "original_id", "chunk"])
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
        greedy_predictions = inference.greedy_predictions(
            classification_probabilities, 
            root=self.classification_tree, 
            threshold=prediction_threshold,
        )

        results_df['greedy_prediction'] = [
            node_lineage_string(node)
            for node in greedy_predictions
        ]

        results_df['accession'] = results_df['original_id'].apply(lambda x: x.split("#")[0])
        def get_original_classification(original_id:str):
            if "#" in original_id:
                return original_id.split("#")[1]
            return "null"
        
        def get_prediction_probability(row):
            prediction = row["greedy_prediction"]
            if prediction in row:
                return row[prediction]
            return 1.0
        
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

        if not (image_dir or output_fasta or output_csv or output_tips_csv):
            print("No output files requested.")

        if output_fasta:
            console.print(f"Writing results for {len(results_df)} repeats to: {output_fasta}")
            output_fasta = Path(output_fasta)
            output_fasta.parent.mkdir(exist_ok=True, parents=True)
            with open(output_fasta, "w") as fasta_out:
                for file in self.dataloader.files:
                    for record in self.dataloader.parse(file):
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

                        if prediction in row:
                            new_probability = row[prediction].values[0]
                        else:
                            new_probability = 1.0 # i.e.root
                        record.description = f"{record.description} original classification = {original_classification}, classification probability = {new_probability:.2f} )"

                        SeqIO.write(record, fasta_out, "fasta")

        if output_tips_csv:
            output_tips_csv = Path(output_tips_csv)
            output_tips_csv.parent.mkdir(exist_ok=True, parents=True)
            non_tips = [node_lineage_string(node) for node in self.classification_tree.node_list if not node.is_leaf]
            tips_df = results_df.drop(columns=non_tips)
            tips_df.to_csv(output_tips_csv, index=False)

        if output_csv:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(exist_ok=True, parents=True)
            console.print(f"Writing results for {len(results_df)} repeats to: {output_csv}")
            results_df.to_csv(output_csv, index=False)

        # if self.vector:
        #     # x = results_df.to_xarray()
        #     # breakpoint()
        #     # embeddings = xr.DataArray(results[0][1], dims=("accession", "embedding"))
        #     # embeddings.to_netcdf("embeddings.nc")
        #     torch.save(results[0][1], "embeddings.pkl")

        return results_df

    def pretrained_location(self) -> str:
        raise NotImplementedError
        return "https://github.com/rbturnbull/terrier/releases/download/v0.1.1-alpha/terrier-0.1.pkl"
    
    @ta.tool
    def create_repeatmasker_seqtree(self, output:Path, repbase:Path, label_smoothing:float=0.0, gamma:float=0.0, partitions:int=5):
        return create_repeatmasker_seqtree(
            output=output,
            repbase=repbase,
            label_smoothing=label_smoothing,
            gamma=gamma,
            partitions=partitions,
        )

    @ta.tool
    def preprocess(
        self, 
        repbase:Path=ta.Param(..., help="The path to the RepBase fasta directory."), 
        seqbank:Path=ta.Param(..., help="The path to save the new SeqBank file."), 
        seqtree:Path=ta.Param(..., help="The path to save the new SeqTree file."), 
        label_smoothing:float=0.0, 
        gamma:float=0.0, 
        partitions:int=5,
    ):
        seqbank = SeqBank(path=seqbank, write=True)
        assert repbase is not None
        repbase = Path(repbase)
        assert repbase.exists()

        # Create the seqbank from the FASTA files with .ref extension
        files = list(repbase.glob('*.ref'))
        seqbank.add_files(files, format="fasta")

        # Create the seqtree
        return create_repeatmasker_seqtree(
            output=seqtree,
            repbase=repbase,
            label_smoothing=label_smoothing,
            gamma=gamma,
            partitions=partitions,
        )

    @ta.tool
    def threshold_plot(
        self, 
        csv:Path = ta.Param(..., help="The CSV file with the results."),
        output:Path=ta.Param(default=None, help="A path to write the confusion matrix, can be HTML or an image file."),
        superfamily:bool=ta.Param(default=True, help="Whether to use the superfamily level for the confusion matrix."),
        show:bool=ta.Param(default=True, help="Whether to show the confusion matrix."),
        width:int=800,
        height:int=800,
        map:str="",
        ignore:str="",
    ) -> pd.DataFrame:
        df = pd.read_csv(csv)
        
        fig = threshold_fig(df, superfamily=superfamily, map=map, ignore=ignore, width=width, height=height)
        if show:
            fig.show()

        if output:
            output = Path(output)
            output.parent.mkdir(exist_ok=True, parents=True)
            print(f"Writing threshold figure to: {output}")
            match output.suffix.lower():
                case ".html":
                    fig.write_html(str(output))
                case _:
                    fig.write_image(str(output), engine="kaleido")


    @ta.tool
    def confusion_matrix(
        self, 
        csv:Path = ta.Param(..., help="The CSV file with the results."),
        output:Path=ta.Param(default=None, help="A path to write the confusion matrix, can be CSV, HTML or an image file."),
        superfamily:bool=ta.Param(default=True, help="Whether to use the superfamily level for the confusion matrix."),
        show:bool=ta.Param(default=True, help="Whether to show the confusion matrix."),
        width:int=800,
        height:int=800,
        map:str="",
        ignore:str="",
        threshold:float=None,
    ) -> pd.DataFrame:
        df = pd.read_csv(csv)
        
        cm = build_confusion_matrix(df, superfamily=superfamily, map=map, ignore=ignore, threshold=threshold)
        fig = confusion_matrix_fig(cm, width=width, height=height)
        if show:
            fig.show()

        if output:
            output = Path(output)
            output.parent.mkdir(exist_ok=True, parents=True)
            print(f"Writing confusion matrix to: {output}")
            match output.suffix.lower():
                case ".csv":
                    cm.to_csv(output)
                case ".html":
                    fig.write_html(str(output))
                case _:
                    fig.write_image(str(output), engine="kaleido")
        
        return cm
