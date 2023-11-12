import typer
from pathlib import Path
from hierarchicalsoftmax import SoftmaxNode
from Bio import SeqIO
from typing import Dict
import csv

from corgi.seqtree import SeqTree #, AlreadyExists

app = typer.Typer()


def create_mapping(file:Path) -> Dict[str, str]:
    repbase_to_repeatmasker = dict()
    with open(file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            repeatmasker_str = row['repeatmasker_type']
            if row['repeatmasker_subtype']:
                repeatmasker_str += "/" + row['repeatmasker_subtype']
            if not repeatmasker_str:
                repeatmasker_str = "Unknown"

            repbase_to_repeatmasker[row['repbase'].strip()] = repeatmasker_str

    return repbase_to_repeatmasker


@app.command()
def create_repbase_seqtree_repeatmasker(output:Path, repbase:Path, label_smoothing:float=0.0, gamma:float=0.0, partitions:int=5):

    mapping = create_mapping(Path(__file__).parent/"data/repbase-to-repeatmasker2.csv")

    # Create Tree From Text File
    classification_tree = SoftmaxNode(name="root", label_smoothing=label_smoothing, gamma=gamma)
    classification_nodes = {}
        
    seqtree = SeqTree(classification_tree)

    # Read files
    count = 0
    for file in repbase.glob('*.ref'):
        with open(file) as f:
            for record in SeqIO.parse(f, "fasta"):
                partition = count % partitions
                accession = record.id

                components = record.description.split("\t")
                if len(components) != 3:
                    if file.name == "simple.ref":
                        classification = "Simple Repeat"
                    elif accession.startswith("SINE_"):
                        classification = "SINE"
                    else:
                        continue
                else:
                    classification = components[1]

                if classification not in mapping:
                    continue

                repeat_name = mapping[classification]
                if repeat_name == "Unknown":
                    continue

                if repeat_name not in classification_nodes:
                    components = repeat_name.split("/")
                    repeat_type = components[0]
                    repeat_subtype = components[1] if len(components) > 1 else ""

                    if repeat_type not in classification_nodes:
                        classification_nodes[repeat_type] = SoftmaxNode(repeat_type, parent=classification_tree, label_smoothing=label_smoothing, gamma=gamma, repeat_masker_name=repeat_type)
                    repeat_type_node = classification_nodes[repeat_type]
                    if repeat_subtype:
                        classification_nodes[repeat_name] = SoftmaxNode(repeat_subtype, parent=repeat_type_node, label_smoothing=label_smoothing, gamma=gamma, repeat_masker_name=repeat_name)

                node = classification_nodes[repeat_name]
            
                try:
                    seqtree.add(accession, node, partition)
                except Exception as err:
                    print(err)

                count += 1

    seqtree.save(output)


if __name__ == "__main__":
    app()
