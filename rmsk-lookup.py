#!/usr/bin/env python3
"""
Cross-references unsupported Repbase annotations against UCSC RepeatMasker (rmsk) tables
to propose mappings for terrier/data/repbase-to-repeatmasker.toml.

Input:
  * the classifications CSV from `terrier-tools preprocess --classifications-csv`
    (columns: provided, repeat_masker, count)
  * one or more rmsk tables (UCSC goldenPath `rmsk.txt.gz`, or a Table Browser download)

For every unsupported `provided` string that exactly matches (case-insensitively) an rmsk
`repName`, the RepeatMasker class and family are looked up and converted to a target that
already exists in the mapping file. Nothing is written to the TOML unless --toml-out is given,
and then only for family-level suggestions that all agree.

Usage:
    python rmsk-lookup.py \\
        --classifications ../repbase/RepBase31.10.fasta/Repbase-seqtree-classifications.csv \\
        --rmsk mm10.rmsk.txt.gz hg38.rmsk.txt.gz \\
        --output rmsk-suggestions.csv \\
        --toml-out rmsk-suggestions.toml
"""
import argparse
import gzip
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import toml

DEFAULT_MAPPING = Path(__file__).parent / "terrier" / "data" / "repbase-to-repeatmasker.toml"

# Columns of the UCSC goldenPath rmsk.txt.gz (no header line)
GOLDENPATH_COLUMNS = [
    "bin", "swScore", "milliDiv", "milliDel", "milliIns", "genoName", "genoStart", "genoEnd",
    "genoLeft", "strand", "repName", "repClass", "repFamily", "repStart", "repEnd", "repLeft", "id",
]

# RepeatMasker (class, family) -> terrier target where the names differ or a prefix rule is not enough.
# Anything not covered here falls back to "<class>/<family prefix>" if that target exists in the mapping.
FAMILY_ALIASES = {
    ("SINE", "Alu"): "SINE/7SL",
    ("SINE", "7SL"): "SINE/7SL",
    ("SINE", "tRNA"): "SINE/tRNA",
    ("SINE", "B2"): "SINE/tRNA",
    ("SINE", "ID"): "SINE/tRNA",
    ("SINE", "MIR"): "SINE/tRNA",
    ("SINE", "5S"): "SINE/5S",
    ("LINE", "Jockey"): "LINE/Jockey-I",
    ("LINE", "I"): "LINE/Jockey-I",
    ("LINE", "I-Jockey"): "LINE/Jockey-I",
    ("LINE", "Jockey-I"): "LINE/Jockey-I",
    ("LINE", "R1-LOA"): "LINE/R1",
    ("LINE", "LOA"): "LINE/R1",
    ("LINE", "Dong-R4"): "LINE/Dong-R4",
    ("LTR", "ERV"): "LTR/ERV",
    ("LTR", "ERV1"): "LTR/ERV",
    ("LTR", "ERVK"): "LTR/ERV",
    ("LTR", "ERVL"): "LTR/ERV",
    ("LTR", "ERVL-MaLR"): "LTR/ERV",
    ("LTR", "BEL"): "LTR/Pao",
    ("DNA", "PIF-Harbinger"): "DNA/Harbinger",
    ("DNA", "MULE-MuDR"): "DNA/MULE",
    ("DNA", "Maverick"): "DNA/Maverick",
    ("RC", "Helitron"): "RC/Helitron",
}

# Used when only the class is trustworthy
CLASS_TARGETS = {
    "SINE": "SINE",
    "LINE": "LINE",
    "LTR": "LTR",
    "DNA": "DNA",
    "Satellite": "Satellite",
    "RNA": "Structural_RNA",
    "rRNA": "Structural_RNA",
    "tRNA": "Structural_RNA",
    "snRNA": "Structural_RNA",
    "scRNA": "Structural_RNA",
}


def read_rmsk(path: Path) -> Counter:
    """Returns counts of (repName, repClass, repFamily) in one rmsk table."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        first = handle.readline()
    has_header = first.lstrip("#").startswith("bin\t") or "repName" in first

    counts = Counter()
    kwargs = dict(sep="\t", chunksize=500_000, dtype=str, low_memory=False)
    if has_header:
        reader = pd.read_csv(path, usecols=lambda c: c.lstrip("#") in {"repName", "repClass", "repFamily"}, **kwargs)
    else:
        reader = pd.read_csv(
            path, header=None, names=GOLDENPATH_COLUMNS, usecols=["repName", "repClass", "repFamily"], **kwargs
        )
    for chunk in reader:
        chunk.columns = [c.lstrip("#") for c in chunk.columns]
        chunk = chunk.fillna("")
        for key, n in chunk.groupby(["repName", "repClass", "repFamily"]).size().items():
            counts[key] += int(n)
    return counts


def suggest(repclass: str, repfamily: str, known_targets: set) -> tuple[str, str]:
    """Returns (target, confidence) where confidence is 'family', 'class' or ''."""
    if repclass.endswith("?") or repfamily.endswith("?"):
        return "", ""  # the curator was unsure

    if (repclass, repfamily) in FAMILY_ALIASES:
        target = FAMILY_ALIASES[(repclass, repfamily)]
        return (target, "family") if target in known_targets else ("", "")

    for candidate in (f"{repclass}/{repfamily}", f"{repclass}/{repfamily.split('-')[0]}"):
        if candidate in known_targets:
            return candidate, "family"

    if repclass in CLASS_TARGETS and CLASS_TARGETS[repclass] in known_targets:
        return CLASS_TARGETS[repclass], "class"
    return "", ""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifications", type=Path, required=True, help="Classifications CSV from terrier-tools preprocess.")
    parser.add_argument("--rmsk", type=Path, nargs="+", required=True, help="UCSC rmsk table(s), plain or .gz.")
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING, help="The current repbase-to-repeatmasker.toml.")
    parser.add_argument("--output", type=Path, default=Path("rmsk-suggestions.csv"))
    parser.add_argument("--toml-out", type=Path, default=None, help="Also write family-level, unambiguous suggestions as TOML lines.")
    args = parser.parse_args()

    mapping = toml.load(args.mapping)
    known_targets = set(mapping.values())

    classifications = pd.read_csv(args.classifications, keep_default_na=False)
    unsupported = classifications[classifications["repeat_masker"] == "<UNSUPPORTED>"]
    print(f"{len(unsupported)} unsupported names covering {unsupported['count'].sum()} sequences")

    # repName (lower case) -> Counter of (repName, repClass, repFamily)
    by_name = defaultdict(Counter)
    for path in args.rmsk:
        print(f"Reading {path}")
        for (name, repclass, repfamily), n in read_rmsk(path).items():
            by_name[name.lower()][(name, repclass, repfamily)] += n

    rows = []
    for provided, count in zip(unsupported["provided"], unsupported["count"]):
        hits = by_name.get(provided.lower())
        if not hits:
            continue
        pairs = Counter()
        for (name, repclass, repfamily), n in hits.items():
            pairs[(repclass, repfamily)] += n

        per_pair = [suggest(c, f, known_targets) for c, f in pairs]
        unique = set(per_pair)
        if len(unique) == 1 and ("", "") not in unique:
            target, confidence = next(iter(unique))
            note = "" if len(pairs) == 1 else "several rmsk (class, family) pairs, same target"
        else:
            target, confidence = "", ""
            note = "ambiguous or unmappable" if len(pairs) > 1 else "no terrier target for this class/family"

        for (repclass, repfamily), n in pairs.most_common():
            rows.append((provided, count, next(iter(hits))[0], repclass, repfamily, n, target, confidence, note))

    columns = ["provided", "count", "rmsk_name", "rmsk_class", "rmsk_family", "rmsk_hits", "suggested_target", "confidence", "note"]
    result = pd.DataFrame(rows, columns=columns)
    result.to_csv(args.output, index=False)

    matched = result["provided"].nunique()
    family_level = result[result["confidence"] == "family"].drop_duplicates("provided")
    class_level = result[result["confidence"] == "class"].drop_duplicates("provided")
    print(f"{matched} unsupported names found in rmsk")
    print(f"  family-level suggestions: {len(family_level)} names, {family_level['count'].sum()} sequences")
    print(f"  class-level suggestions:  {len(class_level)} names, {class_level['count'].sum()} sequences")
    print(f"Wrote {args.output}")

    if args.toml_out:
        with open(args.toml_out, "w") as f:
            for provided, target in zip(family_level["provided"], family_level["suggested_target"]):
                f.write(f'"{provided}" = "{target}"\n')
        print(f"Wrote {args.toml_out} ({len(family_level)} family-level suggestions - review before merging)")


if __name__ == "__main__":
    main()
