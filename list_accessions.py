import typer
from pathlib import Path
from corkie.famdb import FamDB


def main(path: Path):
    famdb = FamDB(path)

    for accession in famdb.get_family_accessions():
        family = famdb.get_family_by_accession(accession)
        if not hasattr(family, "classification") or not family.classification:
            continue
        if family.length < 64:
            continue
    
        print(accession)


if __name__ == "__main__":
    typer.run(main)

