from pathlib import Path
from hierarchicalsoftmax import SoftmaxNode
from Bio import SeqIO
import toml
from corgi.seqtree import SeqTree
from collections import Counter
import gzip


def open_maybe_gz(file:Path):
    if file.name.endswith('.gz'):
        return gzip.open(file, "rt")
    else:
        return open(file, "r")


def get_verbatim_classification(path:Path, record) -> str:
    accession = record.id

    description = record.description

    if "#" in description:
        description = description[ description.find("#")+1 : ]
    
    components = description.split("\t")
    if len(components) >= 2:
        return components[1]

    if path.name == "simple.ref":
        return "Simple Repeat"
    elif accession.startswith("SINE_"):
        return "SINE"
    else:
        return description


def create_repeatmasker_seqtree(
    fasta_paths:list[Path], 
    label_smoothing:float=0.0, 
    gamma:float=0.0, 
    partitions:int=5,
) -> SeqTree:
    with open(Path(__file__).parent/"data/repbase-to-repeatmasker.toml", "r") as f:
        mapping = toml.load(f)

    mapped_counter = Counter()
    not_mapped_counter = Counter()
    
    # Create Tree From Text File
    classification_tree = SoftmaxNode(name="root", label_smoothing=label_smoothing, gamma=gamma)
    classification_nodes = {}
        
    seqtree = SeqTree(classification_tree)

    # Read files
    count = 0
    for file in fasta_paths:
        with open_maybe_gz(file) as f:
            for record in SeqIO.parse(f, "fasta"):
                partition = count % partitions
                accession = record.id

                classification = get_verbatim_classification(file, record)

                if classification in mapping:
                    mapped_counter.update([classification])
                    classification = mapping[classification]

                if classification not in mapping.values():
                    not_mapped_counter.update([classification])
                    continue

                if classification == "Unknown":
                    continue

                if classification not in classification_nodes:
                    components = classification.split("/")
                    repeat_type = components[0]
                    repeat_subtype = components[1] if len(components) > 1 else ""

                    if repeat_type not in classification_nodes:
                        classification_nodes[repeat_type] = SoftmaxNode(
                            repeat_type, 
                            parent=classification_tree, 
                            label_smoothing=label_smoothing, 
                            gamma=gamma, 
                            repeat_masker_name=repeat_type,
                        )
                    repeat_type_node = classification_nodes[repeat_type]
                    if repeat_subtype:
                        classification_nodes[classification] = SoftmaxNode(
                            repeat_subtype, 
                            parent=repeat_type_node, 
                            label_smoothing=label_smoothing, 
                            gamma=gamma, 
                            repeat_masker_name=classification,
                        )

                node = classification_nodes[classification]
            
                try:
                    seqtree.add(accession, node, partition)
                except Exception as err:
                    print(err)

                count += 1

    print("provided,count,mapped,repeat_masker")
    for classification,count in mapped_counter.most_common():
        print(classification,count,1, mapping[classification], sep=",")
    for classification,count in not_mapped_counter.most_common():
        print(classification,count,0, "", sep=",")

    return seqtree

