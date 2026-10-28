from pathlib import Path
from hierarchicalsoftmax import SoftmaxNode
from Bio import SeqIO
import toml
from corgi.seqtree import SeqTree
from collections import Counter, defaultdict
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


def choose_classification(classifications:list[str]) -> str:
    """
    Chooses the classification for an accession which has been given several labels.

    A label is dropped if a more specific label of the same lineage is also present
    (e.g. 'DNA' is dropped in favour of 'DNA/hAT'). If labels from different lineages
    remain (e.g. 'DNA/hAT' and 'DNA/TcMar') then the most specific one is used, then the
    most common one, with ties broken alphabetically. This means that the result does not
    depend on the order of the files, and a generic label never outvotes a specific one.
    """
    counts = Counter(classifications)
    remaining = [
        label for label in counts 
        if not any(other.startswith(f"{label}/") for other in counts)
    ]
    return min(remaining, key=lambda label: (-label.count("/"), -counts[label], label))


def load_accession_rules(mapping:dict[str,str]) -> dict[str,str]:
    """
    Loads the classifications for individual accessions from data/repbase-accession-rules.toml.

    Every classification must be one which the label mapping can produce.
    """
    with open(Path(__file__).parent/"data/repbase-accession-rules.toml", "r") as f:
        rules = toml.load(f)
    invalid = {accession: target for accession, target in rules.items() if target not in mapping.values()}
    if invalid:
        raise ValueError(f"Accession rules with classifications which are not in the mapping: {invalid}")
    return rules


def create_repeatmasker_seqtree(
    fasta_paths:list[Path], 
    label_smoothing:float=0.0, 
    gamma:float=0.0, 
    partitions:int=5,
    seed:int=0,
    summary_csv:Path=None,
    classifications_csv:Path=None,
    retained_csv:Path=None,
    accession_rules:dict[str,str]=None,
) -> SeqTree:
    if partitions < 1:
        raise ValueError("partitions must be a positive integer")

    with open(Path(__file__).parent/"data/repbase-to-repeatmasker.toml", "r") as f:
        mapping = toml.load(f)

    if accession_rules is None:
        accession_rules = load_accession_rules(mapping)

    classification_counts = Counter()
    retained_counts = Counter()
    total_bases = 0
    
    # Create Tree From Text File
    classification_tree = SoftmaxNode(name="root", label_smoothing=label_smoothing, gamma=gamma)
    classification_nodes = {}
        
    seqtree = SeqTree(classification_tree)

    def get_node(classification:str) -> SoftmaxNode:
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
        return classification_nodes[classification]

    # Read files. The same accession can appear in several files with different labels,
    # so collect every labelled occurrence first and only then add each accession once.
    occurrences = defaultdict(list)  # accession -> [(provided, status, classification, digest)]
    for file in fasta_paths:
        with open_maybe_gz(file) as f:
            for record in SeqIO.parse(f, "fasta"):
                accession = record.id

                provided = get_verbatim_classification(file, record)
                classification = mapping.get(provided, provided)
                total_bases += len(record.seq)
                if accession in accession_rules:
                    classification = accession_rules[accession]
                    status = "rule"
                else:
                    if classification == "Unknown":
                        classification_counts[(provided, "unknown", classification)] += 1
                        continue
                    if classification not in mapping.values():
                        classification_counts[(provided, "unsupported", "")] += 1
                        continue
                    status = "mapped" if provided in mapping else "direct"

                get_node(classification)  # creates nodes in the order that labels first appear
                payload = f"{seed}:{str(record.seq).upper()}".encode("utf-8")
                digest = int.from_bytes(hashlib.sha256(payload).digest(), "big")
                occurrences[accession].append((provided, status, classification, digest))

    for accession, entries in occurrences.items():
        classification = choose_classification([entry[2] for entry in entries])
        node = classification_nodes[classification]
        # The partition comes from the sequence so that it does not depend on file order
        partition = min(digest for _, _, label, digest in entries if label == classification) % partitions
        seqtree.add(accession, node, partition)

        for provided, status, label, _ in entries:
            if label == classification:
                retained_counts[label] += 1
                classification_counts[(provided, status, label)] += 1
            elif classification.startswith(f"{label}/"):
                # A more general label for an accession that also has a specific one
                classification_counts[(provided, "superseded", label)] += 1
            else:
                classification_counts[(provided, "add_failed", label)] += 1


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
        ("Accession rules applied", statuses["rule"]),
        ("Unsupported sequences", statuses["unsupported"]),
        ("Generic labels superseded", statuses["superseded"]),
        ("Unknown sequences", statuses["unknown"]),
        ("Conflicting labels dropped", statuses["add_failed"]),
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
