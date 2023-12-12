import typer
from pathlib import Path
from hierarchicalsoftmax import SoftmaxNode
from Bio import SeqIO
from collections import Counter

from corgi.seqtree import SeqTree, AlreadyExists

app = typer.Typer()


@app.command()
def create_repbase_seqtree(output:Path, repbase:Path, label_smoothing:float=0.0, gamma:float=0.0, partitions:int=5):
    # Create Tree From Text File
    classification_tree = SoftmaxNode(name="root", label_smoothing=label_smoothing, gamma=gamma)
    classification_nodes = {}
        
    prev = classification_tree
    parent = classification_tree
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
            classification_nodes[name] = prev
    classification_tree.set_indexes()

    seqtree = SeqTree(classification_tree)

    counter = Counter()

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

                print(classification)
                counter.update([classification])
                if classification not in classification_nodes:
                    continue
                    breakpoint()
                    raise ValueError(f"{classification_tree.render()}\nError: Classification {classification} not in tree above for accession: {accession} in file {file}.\n")
                
                node = classification_nodes[classification]
                
                try:
                    seqtree.add(accession, node, partition)
                except AlreadyExists as err:
                    print(err)

                count += 1

    for key, value in counter.most_common(): 
        print(key, value, sep=",")

    seqtree.save(output)


if __name__ == "__main__":
    app()
