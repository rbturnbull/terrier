import typer
from corgi.seqtree import SeqTree
from pathlib import Path
import json

app = typer.Typer()


@app.command()
def print_name(path:Path):
    seqtree = SeqTree.load(path)

    data = {}

    for accession in seqtree:
        node = seqtree.node(accession)
        data[accession] = getattr(node, 'repeat_masker_name', str(node))

    print(json.dumps(data, indent=2))

if __name__ == "__main__":
    app()
