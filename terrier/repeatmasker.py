from pathlib import Path
from hierarchicalsoftmax import SoftmaxNode
from Bio import SeqIO
import toml
from corgi.seqtree import SeqTree
from collections import Counter


def create_repeatmasker_seqtree(output:Path, repbase:Path, label_smoothing:float=0.0, gamma:float=0.0, partitions:int=5):
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
                    not_mapped_counter.update([classification])
                    continue

                mapped_counter.update([classification])

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

    print("repbase,count,mapped,repeat_masker")
    for classification,count in mapped_counter.most_common():
        print(classification,count,1, mapping[classification], sep=",")
    for classification,count in not_mapped_counter.most_common():
        print(classification,count,0, "", sep=",")

    seqtree.save(output)
    seqtree.classification_tree.render(print=1)


