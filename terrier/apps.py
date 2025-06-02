import pandas as pd
from pathlib import Path
import torchapp as ta
from torchmetrics import Metric
from hierarchicalsoftmax.metrics import RankAccuracyTorchMetric
from hierarchicalsoftmax import inference
import torch
from Bio import SeqIO
from seqbank import SeqBank
from corgi import Corgi
from rich.table import Table
from rich.box import SIMPLE


from rich.console import Console
console = Console()

from .repeatmasker import create_repeatmasker_seqtree
from .evaluate import build_confusion_matrix, confusion_matrix_fig, threshold_fig, evaluate_results, DEFAULT_HEIGHT, DEFAULT_WIDTH, comparison_plot


def output_results_bar_chart(predictions: pd.Series, top_k: int = 10, bar_size: int = 80):
    value_counts = predictions.value_counts()
    total_categories = len(value_counts)
    if top_k:
        original_value_counts = value_counts
        value_counts = value_counts.iloc[:top_k]
    table = Table(box=SIMPLE)
    table.add_column("Prediction", justify="left", style="bold")
    table.add_column("Proportion", justify="left", style="blue")
    table.add_column("Count", justify="right")
    for prediction, count in value_counts.items():
        proportion = count / len(predictions)
        bar = "█" * int(proportion * bar_size)
        table.add_row(prediction, f"{bar} {proportion:.1%}", f"{count}")
    if total_categories > top_k:
        count = sum(original_value_counts.iloc[top_k:])
        proportion = count / len(predictions)
        bar = "█" * int(proportion * bar_size)
        table.add_row(
            "Other Categories",
            f"{bar} {proportion:.1%}",
            f"{count}"
        )
    # Add line
    table.add_row('────────────────────', '─' * bar_size, '─' * 10, style="red")
    table.add_row(
        "Total",
        "",
        f"{len(predictions)}"
    )

    console.print(table)


class Terrier(Corgi):
    """
    Transposable Element Repeat Result classifIER
    """
    @ta.method("super")
    def data(self, **kwargs):
        data = super().data(**kwargs)
        return data
    
    def get_bibtex_files(self) -> list[Path]:
        files = super().get_bibtex_files()
        files.append(Path(__file__).parent / "terrier.bib")
        return files

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
        unknown_only:bool = ta.Param(default=None, help="Whether or not to only rename the 'Unknown' sequences in the FASTA file."),
        image_dir: Path = ta.Param(default=None, help="A directory to output the results as images."),
        image_format:str = "png",
        image_threshold:float = 0.005,
        threshold:float = ta.Param(default=0.7, help="The threshold value for making hierarchical predictions."),
        **kwargs,
    ):        
        def node_lineage_string(node) -> str:
            if node.is_root:
                return "Unknown"
            node_string = "/".join([str(n) for n in node.ancestors[1:]] + [str(node)])

            # An earlier version of Terrier converted all LINE/I to LINE/Jockey-I
            if node_string == "LINE/Jockey-I":
                return "LINE/I"

            return node_string

        classification_probabilities = inference.node_probabilities(results[0], root=self.classification_tree)
        category_names = [node_lineage_string(node) for node in self.classification_tree.node_list if not node.is_root]

        chunk_details = pd.DataFrame(self.dataloader.chunk_details, columns=["file", "original_id", "description", "chunk"])
        predictions_df = pd.DataFrame(classification_probabilities.numpy(), columns=category_names)

        results_df = pd.concat(
            [chunk_details.drop(columns=['chunk']), predictions_df],
            axis=1,
        )

        # Average over chunks
        results_df["order"] = results_df.index
        results_df = results_df.groupby(["file", "original_id", "description"]).mean().reset_index()

        # sort to get original order
        results_df = results_df.sort_values(by="order").drop(columns=["order"])
        
        # results_df['max_leaf_probability_prediction'] = results_df[leaf_names].idxmax(axis=1)

        # Get new tensors now that we've averaged over chunks
        classification_probabilities = torch.as_tensor(results_df[category_names].to_numpy()) 
        # get greedy predictions which can use the raw activation or the softmax probabilities
        greedy_predictions = inference.greedy_predictions(
            classification_probabilities, 
            root=self.classification_tree, 
            threshold=threshold,
        )

        results_df['prediction'] = [
            node_lineage_string(node)
            for node in greedy_predictions
        ]

        results_df['accession'] = results_df['original_id'].apply(lambda x: x.split("#")[0])
        def get_original_classification(original_id:str):
            if "#" in original_id:
                return original_id.split("#")[1]
            return "null"
        
        def get_prediction_probability(row):
            prediction = row["prediction"]
            if prediction in row:
                return row[prediction]
            return 1.0
        
        results_df['probability'] = results_df.apply(get_prediction_probability, axis=1)
        results_df['original_classification'] = results_df['original_id'].apply(get_original_classification)

        # Reorder columns
        results_df = results_df[["file", "accession", "prediction", "probability", "original_id", "original_classification", "description" ] + category_names]

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

        output_results_bar_chart(results_df["prediction"], top_k=10)

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
                        original_classification = row["original_classification"].item().strip()
                        prediction = row["prediction"].item()
                        
                        if not unknown_only or original_classification.lower() == "unknown":
                            record.id = f"{accession}#{prediction}"
                        
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

                        new_description = f"{record.description} original classification = {original_classification}, "

                        if unknown_only:
                            new_description += f"terrier classification = {prediction}, "
                        
                        new_description += f"classification probability = {new_probability:.2f} )"
                        
                        record.description = new_description

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

    def checkpoint(self, checkpoint:Path=None) -> str:
        return checkpoint or "https://github.com/rbturnbull/terrier/releases/download/v0.2.0/terrier-0.2.0.ckpt"
    
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
    def evaluate(
        self, 
        csv:Path = ta.Param(..., help="The CSV file with the results."),
        superfamily:bool=ta.Param(default=True, help="Whether to use the superfamily level for the confusion matrix."),
        map:str="",
        ignore:str="Unknown",
        threshold:float=None,
    ) -> pd.DataFrame:
        df = pd.read_csv(csv)
        return evaluate_results(df, superfamily=superfamily, map=map, ignore=ignore, threshold=threshold)

    @ta.tool
    def threshold_plot(
        self, 
        csv:Path = ta.Param(..., help="The CSV file with the results."),
        output:Path=ta.Param(default=None, help="A path to write the confusion matrix, can be HTML or an image file."),
        superfamily:bool=ta.Param(default=True, help="Whether to use the superfamily level for the confusion matrix."),
        show:bool=ta.Param(default=True, help="Whether to show the confusion matrix."),
        width:int=DEFAULT_WIDTH,
        height:int=DEFAULT_HEIGHT,
        map:str="",
        ignore:str="Unknown",
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
        width:int=DEFAULT_WIDTH,
        height:int=DEFAULT_HEIGHT,
        map:str="",
        ignore:str="Unknown",
        threshold:float=None,
    ) -> pd.DataFrame:
        df = pd.read_csv(csv)
        
        cm = build_confusion_matrix(df, superfamily=superfamily, map=map, ignore=ignore, threshold=threshold)
        fig = confusion_matrix_fig(cm, width=width, height=height, title=f"{csv.name} Confusion Matrix")
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


    @ta.tool
    def comparison_plot(
        self, 
        csv:list[Path] = ta.Param(..., help="The CSV file(s) with the results."),
        output:Path=ta.Param(default=None, help="A path to write the comparison plots, can be HTML or an image file."),
        superfamily:bool=ta.Param(default=True, help="Whether to use the superfamily level for the confusion matrix."),
        show:bool=ta.Param(default=True, help="Whether to show the confusion matrix."),
        threshold:float=None,
    ):
        """ Plot the comparison of Terrier results with the original annotations. """
        
        fig = comparison_plot(csv, superfamily=superfamily, threshold=threshold)
        
        if show:
            fig.show()

        if output:
            output = Path(output)
            output.parent.mkdir(exist_ok=True, parents=True)
            print(f"Writing comparison plot to: {output}")
            match output.suffix.lower():
                case ".html":
                    fig.write_html(str(output))
                case _:
                    fig.write_image(str(output), engine="kaleido")
        
        return fig

    def package_name(self) -> str:
        """
        Returns the name of the package.
        """
        return "bio-terrier"
