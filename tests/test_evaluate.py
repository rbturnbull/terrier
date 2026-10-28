import pytest
from terrier.evaluate import build_map



def test_empty_string():
    # Test case: empty string
    assert build_map("") == {}

def test_single_key_value_pair():
    # Test case: single key-value pair
    assert build_map("key1=value1") == {"key1": "value1"}

def test_multiple_key_value_pairs():
    # Test case: multiple key-value pairs
    assert build_map("key1=value1,key2=value2,key3=value3") == {
        "key1": "value1",
        "key2": "value2",
        "key3": "value3",
    }

def test_key_without_value():
    # Test case: key without a value
    with pytest.raises(IndexError):
        build_map("key1=value1,key2")


def test_whitespace_handling():
    # Test case: whitespace handling
    assert build_map(" key1 = value1 , key2 = value2 ") == {
        " key1 ": " value1 ",
        " key2 ": " value2 ",
    }


from terrier.evaluate import (
    build_str_list,
    map_replace,
    map_replace_list,
    format_fig,
    show_fig,
    evaluate_results,
    evaluate_metrics,
    threshold_fig,
    build_confusion_matrix,
    confusion_matrix_fig,
    comparison_plot,
)
from terrier.defaults import DEFAULT_WIDTH, DEFAULT_HEIGHT
import plotly.graph_objects as go
import numpy as np


def test_build_str_list():
    assert build_str_list("") == []
    assert build_str_list("Unknown") == ["Unknown"]
    assert build_str_list("Unknown, ARTEFACT ,Other") == ["Unknown", "ARTEFACT", "Other"]


def test_map_replace_applies_rules_in_order():
    rules = build_map("hAT-.*=hAT,TcMar-Pogo=TcMar")
    assert map_replace("DNA/hAT-Ac", rules) == "DNA/hAT"
    assert map_replace("DNA/TcMar-Pogo", rules) == "DNA/TcMar"
    assert map_replace("LTR/Gypsy", rules) == "LTR/Gypsy"
    assert map_replace_list(["DNA/hAT-Tip100", "LINE/L1"], rules) == ["DNA/hAT", "LINE/L1"]


def test_benchmark_rules_for_jockey_i():
    rules = build_map("/I-Jockey=/Jockey-I,/Jockey.*=/Jockey-I,/I$=/Jockey-I")
    for label in ["LINE/I-Jockey", "LINE/Jockey", "LINE/I", "LINE/Jockey-I"]:
        assert map_replace(label, rules) == "LINE/Jockey-I"
    assert map_replace("LINE/L1", rules) == "LINE/L1"


def test_show_fig_displays_the_figure(monkeypatch):
    shown = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: shown.append(self))
    fig = go.Figure()
    show_fig(fig)
    assert shown == [fig]


def test_show_fig_handles_missing_display(monkeypatch, capsys):
    def fail(self, *args, **kwargs):
        raise RuntimeError("no display")

    monkeypatch.setattr(go.Figure, "show", fail)
    show_fig(go.Figure())
    output = capsys.readouterr().out
    assert "Could not show figure" in output
    assert "no display" in output


def test_format_fig():
    fig = format_fig(go.Figure())
    assert fig.layout.width == DEFAULT_WIDTH
    assert fig.layout.height == DEFAULT_HEIGHT
    assert fig.layout.plot_bgcolor == "white"


def test_evaluate_results_superfamily(results_df):
    actual, predicted, probability = evaluate_results(results_df, superfamily=True)
    # r4 is ignored because it is Unknown and r5 has no subtype prediction
    assert list(actual) == ["DNA/hAT", "DNA/TcMar", "LTR/Gypsy"]
    assert list(predicted) == ["DNA/hAT", "DNA/hAT", "LTR/Gypsy"]
    assert list(probability) == [0.9, 0.6, 0.95]


def test_evaluate_results_threshold(results_df):
    actual, predicted, probability = evaluate_results(results_df, superfamily=True, threshold=0.7)
    assert list(actual) == ["DNA/hAT", "LTR/Gypsy"]
    assert list(probability) == [0.9, 0.95]


def test_evaluate_results_map(results_df):
    actual, predicted, _ = evaluate_results(results_df, superfamily=True, map="hAT=TcMar")
    assert list(predicted) == ["DNA/TcMar", "DNA/TcMar", "LTR/Gypsy"]
    assert (actual == predicted).sum() == 3


def test_evaluate_results_ignore_list(results_df):
    actual, _, _ = evaluate_results(results_df, superfamily=True, ignore=["Unknown", "DNA/hAT"])
    assert list(actual) == ["DNA/TcMar", "LTR/Gypsy"]


def test_evaluate_results_class_level_uses_probability_columns(results_df):
    actual, predicted, probability = evaluate_results(results_df, superfamily=False)
    assert list(actual) == ["DNA", "DNA", "LTR", "LTR"]
    assert list(predicted) == ["DNA", "DNA", "LTR", "LTR"]
    assert list(probability) == [0.9, 0.7, 0.95, 0.6]


def test_evaluate_results_class_level_without_probability_columns(results_df):
    df = results_df.drop(columns=["DNA", "LTR"])
    actual, predicted, probability = evaluate_results(df, superfamily=False)
    assert list(actual) == ["DNA", "DNA", "LTR", "LTR"]
    assert list(predicted) == ["DNA", "DNA", "LTR", "LTR"]
    assert list(probability) == [0.9, 0.6, 0.95, 0.5]


def test_evaluate_results_without_probability(results_df):
    df = results_df.drop(columns=["probability"])
    _, _, probability = evaluate_results(df, superfamily=True)
    assert list(probability) == [1.0, 1.0, 1.0]


def test_evaluate_metrics(results_df):
    classified, accuracy = evaluate_metrics(results_df, superfamily=True)
    assert classified == pytest.approx(3 / 5)
    assert accuracy == pytest.approx(2 / 3)


def test_threshold_fig(results_df):
    fig = threshold_fig(results_df, superfamily=True, width=500, height=300)
    assert [trace.name for trace in fig.data] == ["Correct", "Classified"]
    assert fig.layout.width == 500
    assert fig.layout.height == 300
    # Sorted by probability so the highest threshold comes first
    assert list(fig.data[0].x) == [0.95, 0.9, 0.6]


def test_build_confusion_matrix(results_df):
    cm = build_confusion_matrix(results_df, superfamily=True)
    assert list(cm.index) == ["DNA/TcMar", "DNA/hAT", "LTR/Gypsy"]
    assert list(cm.columns) == list(cm.index)
    assert cm.to_numpy().sum() == 3
    assert cm.loc["DNA/hAT", "DNA/hAT"] == 1
    assert cm.loc["DNA/TcMar", "DNA/hAT"] == 1
    assert cm.loc["LTR/Gypsy", "LTR/Gypsy"] == 1


def test_confusion_matrix_fig(results_df):
    cm = build_confusion_matrix(results_df, superfamily=True)
    fig = confusion_matrix_fig(cm, width=400, height=300, title="Example")
    assert fig.layout.title.text == "Example"
    assert fig.layout.width == 400
    assert fig.layout.height == 300

    # An empty row does not cause a division by zero
    cm.iloc[0, :] = 0
    confusion_matrix_fig(cm)


@pytest.fixture
def comparison_csvs(results_df, tmp_path):
    paths = []
    for name in ["fruit_fly-families-terrier.csv", "rice.csv"]:
        df = results_df.copy()
        df.loc[0, "original_classification"] = "DNA/hAT?"  # curator was unsure
        path = tmp_path / name
        df.to_csv(path, index=False)
        paths.append(path)
    return paths


@pytest.mark.parametrize("superfamily", [True, False])
@pytest.mark.parametrize("threshold", [0.0, 0.7])
def test_comparison_plot(comparison_csvs, superfamily, threshold):
    fig = comparison_plot(comparison_csvs, superfamily=superfamily, threshold=threshold)
    # five bar traces for each file
    assert len(fig.data) == 5 * len(comparison_csvs)
    assert fig.layout.barmode == "stack"
    assert fig.layout.yaxis.title.text == "Fruit fly"
    assert fig.layout.yaxis3.title.text == "Rice"
    categories = set(fig.data[0].x)
    assert ("DNA" in categories) == (not superfamily)
    assert ("DNA/hAT" in categories) == superfamily
