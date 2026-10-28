from pathlib import Path
import pandas as pd
import pytest
from Bio import SeqIO
from terrier.repeatmasker import get_verbatim_classification, create_repeatmasker_seqtree, choose_classification, load_accession_rules


TEST_DATA_DIR = Path(__file__).parent / "test-data"

def test_get_verbatim_classification_repbase():
    fasta = TEST_DATA_DIR / "repbase.ref"
    expected = [
        "DNA transposon",
        "Loa",
        "SINE",
    ]
    count = sum(1 for line in open(fasta) if line.startswith(">"))
    assert count == len(expected)
    with open(fasta) as f:
        for record, exp in zip(SeqIO.parse(f, "fasta"), expected):
            classification = get_verbatim_classification(fasta, record)
            assert classification == exp


def test_get_verbatim_classification_simple():
    fasta = TEST_DATA_DIR / "simple.ref"
    expected = [
        "Simple Repeat",
        "Simple Repeat",
    ]
    count = sum(1 for line in open(fasta) if line.startswith(">"))
    assert count == len(expected)
    with open(fasta) as f:
        for record, exp in zip(SeqIO.parse(f, "fasta"), expected):
            classification = get_verbatim_classification(fasta, record)
            assert classification == exp


def test_get_verbatim_classification_custom():
    expected = [
        "DNA/Academ",
        "LTR/Caulimovirus",
    ]
    fasta = TEST_DATA_DIR / "custom.fna"
    
    count = sum(1 for line in open(fasta) if line.startswith(">"))
    assert count == len(expected)
    with open(fasta) as f:
        for record, exp in zip(SeqIO.parse(f, "fasta"), expected):
            classification = get_verbatim_classification(fasta, record)
            assert classification == exp


def test_create_repeatmasker_seqtree_repbase():
    fasta = TEST_DATA_DIR / "repbase.ref"
    seqtree = create_repeatmasker_seqtree([fasta])
    assert seqtree.classification_tree is not None
    assert seqtree is not None
    assert len(seqtree) == 3
    assert {"IS905", "BAGGINS1", "SINE_DFSFDFS"} == set(seqtree.keys())
    assert seqtree.classification_tree.render_equal(
        """
        root
        ├── DNA
        ├── LINE
        │   └── R1
        └── SINE
        """
    )

def test_create_repeatmasker_seqtree_repbase_simple():
    seqtree = create_repeatmasker_seqtree([TEST_DATA_DIR / "repbase.ref", TEST_DATA_DIR / "simple.ref"])
    assert seqtree.classification_tree is not None
    assert seqtree is not None
    assert len(seqtree) == 5
    assert {"IS905", "BAGGINS1", "SINE_DFSFDFS", "IJFDKSFF", "GOOGLE"} == set(seqtree.keys())
    assert seqtree.classification_tree.render_equal(
        """
        root
        ├── DNA
        ├── LINE
        │   └── R1
        ├── SINE
        └── Simple_repeat
        """
    )    

def test_create_repeatmasker_seqtree_custom():
    seqtree = create_repeatmasker_seqtree([
        TEST_DATA_DIR / "repbase.ref", 
        TEST_DATA_DIR / "simple.ref", 
        TEST_DATA_DIR / "custom.fna",
    ])
    assert seqtree.classification_tree is not None
    assert seqtree is not None
    assert len(seqtree) == 7
    assert {"IS905", "BAGGINS1", "SINE_DFSFDFS", "IJFDKSFF", "GOOGLE", "I123#DNA/Academ", "XXadsfadsf#LTR/Caulimovirus"} == set(seqtree.keys())
    assert seqtree.classification_tree.render_equal(
        """
        root
        ├── DNA
        │   └── Academ
        ├── LINE
        │   └── R1
        ├── SINE
        ├── Simple_repeat
        └── LTR
            └── Caulimovirus
        """
    )    
    assert seqtree["I123#DNA/Academ"].node.name == "Academ"


def test_partitions_stable_with_reordering_and_new_sequences(tmp_path):
    records = [
        f">seq{i}#DNA/Academ\n{'A' * (i + 1)}CGT\n"
        for i in range(20)
    ]
    original = tmp_path / "original.fasta"
    reordered = tmp_path / "reordered.fasta"
    original.write_text("".join(records))
    reordered.write_text(">extra#DNA/Academ\nTTTTCCCC\n" + "".join(reversed(records)))
    first = create_repeatmasker_seqtree([original], seed=42)
    second = create_repeatmasker_seqtree([reordered], seed=42)
    repeated = create_repeatmasker_seqtree([original], seed=42)
    different_seed = create_repeatmasker_seqtree([original], seed=43)
    for accession in first.keys():
        assert first[accession].partition == second[accession].partition
        assert first[accession].partition == repeated[accession].partition
        assert 0 <= first[accession].partition < 5
    assert any(
        first[key].partition != different_seed[key].partition for key in first.keys()
    )


def test_identical_sequences_share_partition(tmp_path):
    fasta = tmp_path / "duplicates.fasta"
    fasta.write_text(">first#DNA/Academ\nACGT\n>second#DNA/Academ\nacgt\n")
    seqtree = create_repeatmasker_seqtree([fasta], seed=42, partitions=3)
    assert seqtree["first#DNA/Academ"].partition == seqtree["second#DNA/Academ"].partition
    assert 0 <= seqtree["first#DNA/Academ"].partition < 3


def test_invalid_partition_count():
    import pytest

    with pytest.raises(ValueError, match="positive integer"):
        create_repeatmasker_seqtree([], partitions=0)


def test_complete_summary_and_csv_exports(tmp_path, capsys):
    import pandas as pd

    fasta = tmp_path / "custom.fasta"
    fasta.write_text(
        ">mapped#LTR\nACGT\n"
        ">direct#LTR/Gypsy\nACGTAC\n"
        ">rejected#DNA/UnsupportedTestClass\nACG\n"
    )
    summary_csv = tmp_path / "reports" / "summary.csv"
    classifications_csv = tmp_path / "classifications.csv"
    retained_csv = tmp_path / "retained.csv"
    tree = create_repeatmasker_seqtree(
        [fasta], summary_csv=summary_csv,
        classifications_csv=classifications_csv, retained_csv=retained_csv,
    )
    assert len(tree) == 2
    summary = pd.read_csv(summary_csv).set_index("metric")["value"]
    assert summary["Input sequences"] == 3
    assert summary["Input bases"] == 13
    assert summary["Retained sequences"] == 2
    assert summary["Unsupported sequences"] == 1
    classes = pd.read_csv(classifications_csv).set_index("provided")
    assert list(classes.reset_index().columns) == ["provided", "repeat_masker", "count"]
    assert classes.loc["LTR", "repeat_masker"] == "LTR"
    assert classes.loc["LTR/Gypsy", "repeat_masker"] == "LTR/Gypsy"
    assert classes.loc["DNA/UnsupportedTestClass", "repeat_masker"] == "<UNSUPPORTED>"
    assert classes["count"].sum() == 3
    retained = pd.read_csv(retained_csv)
    assert retained["count"].sum() == 2
    assert retained["percent_retained"].sum() == 100
    output = capsys.readouterr().out
    assert "Dataset summary" in output
    assert "Input classifications" in output


def test_empty_summary_exports(tmp_path):
    import pandas as pd

    output = tmp_path / "retained.csv"
    tree = create_repeatmasker_seqtree([], retained_csv=output)
    assert len(tree) == 0
    assert list(pd.read_csv(output).columns) == ["repeat_masker", "count", "percent_retained"]


def test_choose_classification():
    assert choose_classification(["DNA"]) == "DNA"
    assert choose_classification(["DNA", "DNA/hAT"]) == "DNA/hAT"
    assert choose_classification(["DNA/hAT", "DNA"]) == "DNA/hAT"
    assert choose_classification(["DNA", "DNA/hAT", "DNA/hAT"]) == "DNA/hAT"
    assert choose_classification(["LTR", "DNA/hAT"]) == "DNA/hAT"  # different lineages
    # A generic label does not outvote a specific one from another lineage
    assert choose_classification(["LINE", "LINE", "LINE", "LTR/ERV"]) == "LTR/ERV"
    assert choose_classification(["DNA", "DNA", "RC/Helitron"]) == "RC/Helitron"
    # Conflicts use the most common label, then alphabetical order
    assert choose_classification(["DNA/hAT", "DNA/TcMar", "DNA/TcMar"]) == "DNA/TcMar"
    assert choose_classification(["DNA/TcMar", "DNA/hAT"]) == "DNA/TcMar"


def test_most_specific_label_wins_for_repeated_accession(tmp_path):
    # Repbase headers are 'accession<TAB>classification<TAB>species'
    first = tmp_path / "first.fasta"
    second = tmp_path / "second.fasta"
    first.write_text(">same\tDNA transposon\tHomo sapiens\nACGTACGT\n>other\tDNA transposon\tHomo sapiens\nAACCGGTT\n")
    second.write_text(">same\thAT\tHomo sapiens\nACGTACGT\n")
    for paths in ([first, second], [second, first]):
        seqtree = create_repeatmasker_seqtree(paths, seed=1)
        assert seqtree["same"].node.name == "hAT"
        assert seqtree["other"].node.name == "DNA"


def test_conflicting_labels_do_not_depend_on_file_order(tmp_path):
    first = tmp_path / "first.fasta"
    second = tmp_path / "second.fasta"
    first.write_text(">same\thAT\tHomo sapiens\nACGTACGT\n")
    second.write_text(">same\tMariner/Tc1\tHomo sapiens\nACGTACGT\n")
    results = []
    for paths in ([first, second], [second, first]):
        seqtree = create_repeatmasker_seqtree(paths, seed=1)
        assert len(seqtree) == 1
        results.append(seqtree["same"].node.name)
    assert results[0] == results[1]


def test_repeatmasker_family_names_are_collapsed(tmp_path):
    fasta = tmp_path / "families.fasta"
    fasta.write_text(
        ">a#LINE/I-Jockey\nACGTAC\n>b#LINE/Jockey\nACGTAG\n>c#LINE/I\nACGTAT\n>d#LINE/Jockey-I\nACGTCC\n"
        ">e#DNA/hAT-Ac\nACGTGG\n>f#LINE/R2-Hero\nACGTTT\n"
    )
    seqtree = create_repeatmasker_seqtree([fasta])
    for accession in ["a#LINE/I-Jockey", "b#LINE/Jockey", "c#LINE/I", "d#LINE/Jockey-I"]:
        assert seqtree[accession].node.name == "Jockey-I"
    assert seqtree["e#DNA/hAT-Ac"].node.name == "hAT"
    assert seqtree["f#LINE/R2-Hero"].node.name == "R2"


def test_accession_rules_override_the_label(tmp_path):
    first = tmp_path / "first.fasta"
    second = tmp_path / "second.fasta"
    first.write_text(">ruled\tDNA transposon\tHomo sapiens\nACGTACGT\n>other\tDNA transposon\tHomo sapiens\nAACCGGTT\n")
    second.write_text(">ruled\tMariner/Tc1\tHomo sapiens\nACGTACGT\n")
    for paths in ([first, second], [second, first]):
        seqtree = create_repeatmasker_seqtree(paths, accession_rules={"ruled": "DNA/hAT"})
        assert seqtree["ruled"].node.name == "hAT"  # even though one file says Mariner/Tc1
        assert seqtree["other"].node.name == "DNA"


def test_accession_rules_in_the_package_are_valid():
    import toml
    mapping = toml.load(Path(__file__).parent.parent / "terrier/data/repbase-to-repeatmasker.toml")
    rules = load_accession_rules(mapping)
    assert len(rules) > 0
    assert set(rules.values()) <= set(mapping.values())
    assert rules["MER103"] == "DNA/hAT"


def test_gzipped_fasta_files_are_read():
    plain = create_repeatmasker_seqtree([TEST_DATA_DIR / "custom.fna"])
    zipped = create_repeatmasker_seqtree([TEST_DATA_DIR / "custom.fna.gz"])
    assert set(zipped.keys()) == set(plain.keys())
    assert len(zipped) > 0


def test_accession_rules_must_use_classifications_from_the_mapping():
    with pytest.raises(ValueError, match="not in the mapping"):
        load_accession_rules({"hAT": "DNA/hAT"})


def test_partitions_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        create_repeatmasker_seqtree([TEST_DATA_DIR / "repbase.ref"], partitions=0)


def test_unknown_and_unsupported_labels_are_not_added(tmp_path):
    fasta = tmp_path / "labels.fasta"
    fasta.write_text(">a#Unknown\nACGT\n>b#Not a real class\nACGT\n>c#DNA/hAT\nACGT\n")
    summary_csv = tmp_path / "summary.csv"
    seqtree = create_repeatmasker_seqtree([fasta], summary_csv=summary_csv)
    assert list(seqtree.keys()) == ["c#DNA/hAT"]

    summary = pd.read_csv(summary_csv).set_index("metric")["value"]
    assert summary["Unknown sequences"] == 1
    assert summary["Unsupported sequences"] == 1
    assert summary["Retained sequences"] == 1


def test_summary_describes_how_labels_were_resolved(tmp_path):
    first = tmp_path / "first.fasta"
    second = tmp_path / "second.fasta"
    # Repbase headers are 'accession<TAB>classification<TAB>species'
    first.write_text(
        ">a\tDNA transposon\tx\nACGT\n"   # generic label which is replaced by a specific one
        ">b\tDNA transposon\tx\nACGA\n"   # classified by an accession rule
        ">c\tNot a real class\tx\nACGC\n"  # unsupported
        ">d\tUnknown\tx\nACGG\n"          # unknown
        ">e\thAT\tx\nACGTT\n"             # conflicts with the label in the second file
    )
    second.write_text(">a\thAT\tx\nACGT\n>e\tMariner/Tc1\tx\nACGTT\n")
    summary_csv = tmp_path / "summary.csv"
    classifications_csv = tmp_path / "classifications.csv"
    retained_csv = tmp_path / "retained.csv"

    seqtree = create_repeatmasker_seqtree(
        [first, second],
        accession_rules={"b": "DNA/Academ"},
        summary_csv=summary_csv,
        classifications_csv=classifications_csv,
        retained_csv=retained_csv,
    )

    assert sorted(seqtree.keys()) == ["a", "b", "e"]
    assert seqtree["a"].node.name == "hAT"
    assert seqtree["b"].node.name == "Academ"
    assert seqtree["e"].node.name == "TcMar"  # tie between lineages is broken alphabetically

    summary = pd.read_csv(summary_csv).set_index("metric")["value"]
    assert summary["Input files"] == 2
    assert summary["Input sequences"] == 7
    assert summary["Accession rules applied"] == 1
    assert summary["Unsupported sequences"] == 1
    assert summary["Unknown sequences"] == 1
    assert summary["Generic labels superseded"] == 1
    assert summary["Conflicting labels dropped"] == 1
    assert summary["Retained sequences"] == 3
    assert summary["Retained classifications"] == 3

    classifications = pd.read_csv(classifications_csv).set_index("provided")
    assert classifications.loc["hAT", "repeat_masker"] == "DNA/hAT"
    assert classifications.loc["Not a real class", "repeat_masker"] == "<UNSUPPORTED>"

    retained = pd.read_csv(retained_csv).set_index("repeat_masker")
    assert set(retained.index) == {"DNA/hAT", "DNA/Academ", "DNA/TcMar"}
    assert retained["percent_retained"].sum() == pytest.approx(100.0, abs=0.05)  # each is rounded
