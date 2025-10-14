from pathlib import Path
from Bio import SeqIO
from terrier.repeatmasker import get_verbatim_classification, create_repeatmasker_seqtree


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
        └── Satellite
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
        ├── Satellite
        └── LTR
            └── Caulimovirus
        """
    )    
    assert seqtree["I123#DNA/Academ"].node.name == "Academ"
