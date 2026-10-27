from pathlib import Path
from hierarchicalsoftmax import SoftmaxNode
from Bio import SeqIO
import toml
from corgi.seqtree import SeqTree
from collections import Counter
import gzip
import hashlib
import pandas as pd
from rich.console import Console
from rich.table import Table


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
    seed:int=0,
    summary_csv:Path=None,
    classifications_csv:Path=None,
    retained_csv:Path=None,
) -> SeqTree:
    if partitions < 1:
        raise ValueError("partitions must be a positive integer")

    with open(Path(__file__).parent/"data/repbase-to-repeatmasker.toml", "r") as f:
        mapping = toml.load(f)

    classification_counts = Counter()
    retained_counts = Counter()
    total_bases = 0
    
    # Create Tree From Text File
    classification_tree = SoftmaxNode(name="root", label_smoothing=label_smoothing, gamma=gamma)
    classification_nodes = {}
        
    seqtree = SeqTree(classification_tree)

    # Read files
    for file in fasta_paths:
        with open_maybe_gz(file) as f:
            for record in SeqIO.parse(f, "fasta"):
                accession = record.id

                provided = get_verbatim_classification(file, record)
                classification = mapping.get(provided, provided)
                total_bases += len(record.seq)
                if classification == "Unknown":
                    classification_counts[(provided, "unknown", classification)] += 1
                    continue
                if classification not in mapping.values():
                    classification_counts[(provided, "unsupported", "")] += 1
                    continue
                status = "mapped" if provided in mapping else "direct"


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
                payload = f"{seed}:{str(record.seq).upper()}".encode("utf-8")
                digest = hashlib.sha256(payload).digest()
                partition = int.from_bytes(digest, "big") % partitions
            
                try:
                    seqtree.add(accession, node, partition)
                    retained_counts[classification] += 1
                    classification_counts[(provided, status, classification)] += 1
                except Exception as err:
                    classification_counts[(provided, "add_failed", classification)] += 1
                    print(err)


    input_counts = Counter()
    for (provided, _, _), count in classification_counts.items():
        input_counts[provided] += count
    rows = []
    for provided, count in sorted(input_counts.items(), key=lambda item: (-item[1], item[0])):
        target = mapping.get(provided, provided)
        if target not in mapping.values() or target == "Unknown":
            target = "<UNSUPPORTED>"
        rows.append((provided, target, count))
    classifications = pd.DataFrame(rows, columns=["provided", "repeat_masker", "count"])
    total = sum(classification_counts.values())
    accepted = sum(retained_counts.values())
    statuses = Counter()
    for (_, status, _), count in classification_counts.items():
        statuses[status] += count
    summary = pd.DataFrame([
        ("Input files", len(fasta_paths)),
        ("Input sequences", total),
        ("Input bases", total_bases),
        ("Retained sequences", accepted),
        ("Unsupported sequences", statuses["unsupported"]),
        ("Unknown sequences", statuses["unknown"]),
        ("Failed additions", statuses["add_failed"]),
        ("Retained classifications", len(retained_counts)),
    ], columns=["metric", "value"])
    retained = pd.DataFrame([
        (classification, count, round(100 * count / accepted, 2))
        for classification, count in sorted(retained_counts.items(), key=lambda item: (-item[1], item[0]))
    ], columns=["repeat_masker", "count", "percent_retained"])
    console = Console()
    for title, frame, output in [
        ("Dataset summary", summary, summary_csv),
        ("Input classifications", classifications, classifications_csv),
        ("Retained classifications (direct assignments)", retained, retained_csv),
    ]:
        table = Table(title=title)
        for column in frame.columns:
            table.add_column(column, justify="right" if column in {"count", "value", "percent_retained"} else "left")
        for row in frame.itertuples(index=False, name=None):
            table.add_row(*(str(value) for value in row))
        console.print(table)
        if output is not None:
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output, index=False)
    console.print("<UNSUPPORTED> = no supported classification. SeqBank contains the input sequences.")
    return seqtree
