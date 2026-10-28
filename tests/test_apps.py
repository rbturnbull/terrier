from pathlib import Path
import shutil
import pandas as pd
import pytest
import torch
from Bio import SeqIO
from hierarchicalsoftmax import SoftmaxNode
from typer.testing import CliRunner

from terrier.apps import Terrier
from terrier.repeatmasker import create_repeatmasker_seqtree
from terrier.defaults import DEFAULT_HEIGHT, DEFAULT_WIDTH

from .conftest import TEST_DATA_DIR


INFERENCE_FASTA = """\
>r1#Unknown first (note)
ACGT
>r2#LTR/Gypsy second
ACGT
>r3 third
ACGT
>r4#Unknown extra
ACGT
"""

# Every parameter of output_results is passed because the defaults are torchapp parameter objects
NO_OUTPUT = dict(
    output_csv=None,
    output_tips_csv=None,
    output_fasta=None,
    unknown_only=False,
    image_dir=None,
    image_format="png",
    image_threshold=0.005,
    threshold=0.0,
)


class FakeDataLoader:
    """Stands in for the inference dataloader, which splits sequences into chunks."""

    def __init__(self, files, chunk_ids):
        self.files = files
        self.chunk_details = [(str(files[0]), original_id, description, 0) for original_id, description in chunk_ids]

    def parse(self, file):
        return SeqIO.parse(file, "fasta")


@pytest.fixture
def inference_fasta(tmp_path) -> Path:
    path = tmp_path / "inference.fasta"
    path.write_text(INFERENCE_FASTA)
    return path


@pytest.fixture
def classification_tree() -> SoftmaxNode:
    seqtree = create_repeatmasker_seqtree(
        [TEST_DATA_DIR / "repbase.ref", TEST_DATA_DIR / "simple.ref", TEST_DATA_DIR / "custom.fna"]
    )
    tree = seqtree.classification_tree
    tree.layer_size  # sets the indexes of the nodes
    return tree


@pytest.fixture
def inference_app(classification_tree, inference_fasta) -> Terrier:
    """An app ready to make output from results, with r4 absent from the results. r1 has two chunks."""
    app = Terrier()
    app.classification_tree = classification_tree
    app.dataloader = FakeDataLoader(
        [inference_fasta],
        [
            ("r1#Unknown", "r1#Unknown first (note)"),
            ("r1#Unknown", "r1#Unknown first (note)"),
            ("r2#LTR/Gypsy", "r2#LTR/Gypsy second"),
            ("r3", "r3 third"),
        ],
    )
    return app


def make_results(app, scale=3.0, seed=0):
    torch.manual_seed(seed)
    return [torch.randn(len(app.dataloader.chunk_details), app.classification_tree.layer_size) * scale]


def test_simple_attributes():
    app = Terrier()
    assert app.package_name() == "bio-terrier"
    assert app.monitor() == "accuracy_repeatmasker_subtype"
    assert app.checkpoint().endswith("terrier-0.4.ckpt")
    assert app.checkpoint(Path("custom.ckpt")) == Path("custom.ckpt")
    bibtex = app.get_bibtex_files()
    assert bibtex[-1].name == "terrier.bib"
    assert bibtex[-1].exists()


def test_metrics(classification_tree):
    app = Terrier()
    app.classification_tree = classification_tree
    metrics = app.metrics()
    assert [name for name, _ in metrics] == ["rank_accuracy"]


def test_preprocess_creates_outputs(tmp_path):
    seqtree = tmp_path / "out.st"
    result = CliRunner().invoke(
        Terrier().tools_app,
        [
            "preprocess",
            "--input", str(TEST_DATA_DIR / "repbase.ref"),
            "--input", str(TEST_DATA_DIR / "custom.fna"),
            "--seqbank", str(tmp_path / "out.sb"),
            "--seqtree", str(seqtree),
            "--summary-csv", str(tmp_path / "summary.csv"),
            "--classifications-csv", str(tmp_path / "classifications.csv"),
            "--retained-csv", str(tmp_path / "retained.csv"),
            "--partitions", "2",
        ],
    )
    assert result.exit_code == 0, result.output
    for name in ["out.sb", "out.st", "summary.csv", "classifications.csv", "retained.csv"]:
        assert (tmp_path / name).exists()

    summary = pd.read_csv(tmp_path / "summary.csv").set_index("metric")["value"]
    assert summary["Input files"] == 2
    assert summary["Input sequences"] == 5


def test_preprocess_with_only_a_summary(tmp_path):
    result = CliRunner().invoke(
        Terrier().tools_app,
        ["preprocess", "--input", str(TEST_DATA_DIR / "repbase.ref"), "--summary-csv", str(tmp_path / "summary.csv")],
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "summary.csv").exists()
    assert not (tmp_path / "out.st").exists()


def test_preprocess_requires_an_output():
    result = CliRunner().invoke(Terrier().tools_app, ["preprocess", "--input", str(TEST_DATA_DIR / "repbase.ref")])
    assert result.exit_code != 0
    assert "Provide a --seqbank" in str(result.exception)


def test_preprocess_with_only_empty_files(tmp_path):
    empty = tmp_path / "empty.fasta"
    empty.write_text("")
    result = CliRunner().invoke(
        Terrier().tools_app, ["preprocess", "--input", str(empty), "--seqtree", str(tmp_path / "out.st")]
    )
    assert result.exit_code != 0
    assert "No valid FASTA files" in str(result.exception)


def test_data_from_preprocessed_files(tmp_path):
    result = CliRunner().invoke(
        Terrier().tools_app,
        [
            "preprocess",
            "--input", str(TEST_DATA_DIR / "repbase.ref"),
            "--input", str(TEST_DATA_DIR / "custom.fna"),
            "--seqbank", str(tmp_path / "out.sb"),
            "--seqtree", str(tmp_path / "out.st"),
        ],
    )
    assert result.exit_code == 0, result.output

    data = Terrier().data(seqtree=tmp_path / "out.st", seqbank=tmp_path / "out.sb", batch_size=2, num_workers=0)
    data.setup()
    assert len(data.train_dataset) > 0
    assert len(data.val_dataset) > 0


def test_output_results_returns_a_row_per_sequence(inference_app):
    df = inference_app.output_results(make_results(inference_app), **NO_OUTPUT)
    # r1 has two chunks which are averaged
    assert list(df["accession"]) == ["r1", "r2", "r3"]
    assert list(df["original_classification"]) == ["Unknown", "LTR/Gypsy", "null"]
    assert list(df.columns[:7]) == [
        "file", "accession", "prediction", "probability", "original_id", "original_classification", "description",
    ]
    # A threshold of zero always reaches a tip of the tree
    tips = {
        "/".join([str(n) for n in node.ancestors[1:]] + [str(node)])
        for node in inference_app.classification_tree.node_list
        if node.is_leaf
    }
    assert set(df["prediction"]) <= tips
    assert df["probability"].between(0, 1).all()


def test_output_results_with_a_high_threshold_predicts_the_root(inference_app):
    df = inference_app.output_results(make_results(inference_app), **{**NO_OUTPUT, "threshold": 1.5})
    assert set(df["prediction"]) == {"Unknown"}
    assert list(df["probability"]) == [1.0, 1.0, 1.0]


def test_output_results_without_outputs_says_so(inference_app, capsys):
    inference_app.output_results(make_results(inference_app), **NO_OUTPUT)
    assert "No output files requested." in capsys.readouterr().out


def test_output_results_csv_files(inference_app, tmp_path):
    csv_path = tmp_path / "nested" / "results.csv"
    tips_path = tmp_path / "nested" / "tips.csv"
    df = inference_app.output_results(
        make_results(inference_app), **{**NO_OUTPUT, "output_csv": csv_path, "output_tips_csv": tips_path}
    )
    full = pd.read_csv(csv_path)
    assert list(full["accession"]) == ["r1", "r2", "r3"]
    assert list(full.columns) == list(df.columns)

    # The tips file leaves out the probabilities for the internal nodes
    tips = pd.read_csv(tips_path)
    assert {"SINE", "Simple_repeat"} <= set(tips.columns)
    assert not {"DNA", "LINE", "LTR"} & set(tips.columns)
    assert len(tips) == 3


def test_output_results_fasta_renames_sequences(inference_app, inference_fasta, tmp_path):
    output = tmp_path / "out" / "classified.fasta"
    df = inference_app.output_results(make_results(inference_app), **{**NO_OUTPUT, "output_fasta": output})
    records = {r.id: r for r in SeqIO.parse(output, "fasta")}
    predictions = dict(zip(df["accession"], df["prediction"]))

    # All the classified sequences have their id changed to the accession and prediction
    assert set(records) == {f"r1#{predictions['r1']}", f"r2#{predictions['r2']}", f"r3#{predictions['r3']}", "r4#Unknown"}
    assert "original classification = Unknown" in records[f"r1#{predictions['r1']}"].description
    assert "original classification = LTR/Gypsy" in records[f"r2#{predictions['r2']}"].description
    assert "original classification = null" in records[f"r3#{predictions['r3']}"].description
    assert "classification probability = " in records[f"r1#{predictions['r1']}"].description
    assert "terrier classification" not in records[f"r1#{predictions['r1']}"].description

    # r4 was not in the results so it is written as it was
    assert records["r4#Unknown"].description == "r4#Unknown extra"
    assert str(records["r4#Unknown"].seq) == "ACGT"


def test_output_results_fasta_for_the_root_prediction(inference_app, tmp_path):
    output = tmp_path / "classified.fasta"
    inference_app.output_results(make_results(inference_app), **{**NO_OUTPUT, "output_fasta": output, "threshold": 1.5})
    records = {r.id: r for r in SeqIO.parse(output, "fasta")}
    assert set(records) == {"r1#Unknown", "r2#Unknown", "r3#Unknown"} | {"r4#Unknown"}
    assert "classification probability = 1.00" in records["r1#Unknown"].description


def test_output_results_fasta_unknown_only(inference_app, tmp_path):
    output = tmp_path / "classified.fasta"
    df = inference_app.output_results(
        make_results(inference_app), **{**NO_OUTPUT, "output_fasta": output, "unknown_only": True}
    )
    predictions = dict(zip(df["accession"], df["prediction"]))
    records = {r.id: r for r in SeqIO.parse(output, "fasta")}

    # Only the sequence which was originally Unknown is renamed
    assert f"r1#{predictions['r1']}" in records
    assert "r2#LTR/Gypsy" in records
    assert "r3" in records
    assert f"terrier classification = {predictions['r1']}" in records[f"r1#{predictions['r1']}"].description
    assert "terrier classification = " in records["r2#LTR/Gypsy"].description


@pytest.mark.skipif(shutil.which("dot") is None, reason="Graphviz is needed to draw the probability images")
def test_output_results_images(inference_app, tmp_path):
    image_dir = tmp_path / "images"
    inference_app.output_results(make_results(inference_app), **{**NO_OUTPUT, "image_dir": image_dir})
    directory = image_dir / "inference.fasta"
    assert sorted(p.name for p in directory.iterdir()) == ["r1.png", "r2.png", "r3.png"]


def test_output_results_for_trees_without_a_softmax_node_list(tmp_path):
    """Seqtree files from Terrier 0.3 or earlier use every non-root node as a category."""
    tree = SoftmaxNode("root")
    for name in ["A", "B", "C"]:
        SoftmaxNode(name, parent=tree)
    tree.layer_size  # sets the indexes of the nodes
    del tree.node_list_softmax

    fasta = tmp_path / "old.fasta"
    fasta.write_text(">s1\nACGT\n>s2\nACGT\n")
    app = Terrier()
    app.classification_tree = tree
    app.dataloader = FakeDataLoader([fasta], [("s1", "s1"), ("s2", "s2")])

    df = app.output_results(make_results(app), **{**NO_OUTPUT, "threshold": 0.0})
    assert list(df.columns[-3:]) == ["A", "B", "C"]
    assert set(df["prediction"]) <= {"A", "B", "C"}


def test_evaluate_tool(results_csv):
    actual, predicted, probability = Terrier().evaluate(
        csv=results_csv, superfamily=True, map="", ignore="Unknown", threshold=None
    )
    assert list(actual) == ["DNA/hAT", "DNA/TcMar", "LTR/Gypsy"]
    assert list(predicted) == ["DNA/hAT", "DNA/hAT", "LTR/Gypsy"]


@pytest.mark.parametrize("show", [True, False])
@pytest.mark.parametrize("suffix", [".html", ".png", None])
def test_threshold_plot_tool(results_csv, tmp_path, monkeypatch, stub_image_export, show, suffix):
    shown = []
    monkeypatch.setattr("terrier.evaluate.show_fig", lambda fig: shown.append(fig))
    output = tmp_path / "plots" / f"threshold{suffix}" if suffix else None

    Terrier().threshold_plot(
        csv=results_csv, output=output, superfamily=True, show=show,
        width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, map="", ignore="Unknown",
    )
    assert len(shown) == (1 if show else 0)
    if output:
        assert output.exists()
        if suffix == ".html":
            assert "plotly" in output.read_text()


@pytest.mark.parametrize("show", [True, False])
@pytest.mark.parametrize("suffix", [".csv", ".html", ".png", None])
def test_confusion_matrix_tool(results_csv, tmp_path, monkeypatch, stub_image_export, show, suffix):
    shown = []
    monkeypatch.setattr("terrier.evaluate.show_fig", lambda fig: shown.append(fig))
    output = tmp_path / "matrices" / f"matrix{suffix}" if suffix else None

    cm = Terrier().confusion_matrix(
        csv=results_csv, output=output, superfamily=True, show=show,
        width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT, map="", ignore="Unknown", threshold=None,
    )
    assert cm.to_numpy().sum() == 3
    assert len(shown) == (1 if show else 0)
    if output:
        assert output.exists()
        if suffix == ".csv":
            assert pd.read_csv(output, index_col=0).shape == (3, 3)


@pytest.mark.parametrize("show", [True, False])
@pytest.mark.parametrize("suffix", [".html", ".png", None])
def test_comparison_plot_tool(results_csv, tmp_path, monkeypatch, stub_image_export, show, suffix):
    shown = []
    monkeypatch.setattr("terrier.evaluate.show_fig", lambda fig: shown.append(fig))
    output = tmp_path / "comparisons" / f"comparison{suffix}" if suffix else None

    fig = Terrier().comparison_plot(csv=[results_csv], output=output, superfamily=True, show=show, threshold=0.7)
    assert len(fig.data) == 5
    assert len(shown) == (1 if show else 0)
    if output:
        assert output.exists()


def test_comparison_plot_tool_without_a_threshold(results_csv):
    fig = Terrier().comparison_plot(csv=[results_csv], output=None, superfamily=False, show=False, threshold=None)
    assert len(fig.data) == 5
