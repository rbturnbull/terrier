import pandas as pd
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import plotly.io as pio   
pio.kaleido.scope.mathjax = None


def build_map(map:str) -> dict[str,str]:
    """
    Builds a dictionary from a string in the form of 'key1=value1|key2=value2'
    """
    components = map.split(",")
    return {x.split("=")[0]: x.split("=")[1] for x in components}


def build_str_list(string:str) -> list[str]:
    """
    Builds a dictionary from a string in the form of 'key1=value1|key2=value2'
    """
    return [x.strip() for x in string.split(",")]


def map_replace(string:str, map:dict[str,str]) -> str:
    """
    Replaces values in a string using a map.
    """
    for key, value in map.items():
        string = string.replace(key, value)
    return string


def map_replace_list(iterable:list[str], map:dict[str,str]) -> list[str]:
    """
    Replaces values in an iterable using a map.
    """
    return [map_replace(x, map) for x in iterable]


def format_fig(fig):
    """Formats a plotly figure in a nicer way."""
    fig.update_layout(
        width=1200,
        height=550,
        plot_bgcolor="white",
        title_font_color="black",
        font=dict(
            family="Linux Libertine Display O",
            size=18,
            color="black",
        ),
    )
    gridcolor = "#dddddd"
    fig.update_xaxes(gridcolor=gridcolor)
    fig.update_yaxes(gridcolor=gridcolor)

    fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside')
    fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True, ticks='outside')

    return fig


def build_confusion_matrix(data:pd.DataFrame, superfamily:bool=True, map:str|dict="", ignore:str|list[str]="", threshold:float|None=None) -> pd.DataFrame:
    if isinstance(map, str):
        map = build_map(map)
    
    if isinstance(ignore, str):
        ignore = build_str_list(ignore)

    total = len(data)
    print(f"Total: {total}")

    # Remove ignored classes
    data = data[~data['original_classification'].isin(ignore)]

    # Filter ground truth
    ground_truth_total = len(data)
    print(f"Total with ground truth: {ground_truth_total}")

    # Filter predictions
    if threshold is not None:
        data = data[data['probability'] >= threshold]
    data = data[~data['greedy_prediction'].isin(ignore)]
    prediction_total = len(data)
    print(f"Number classified: {prediction_total}/{ground_truth_total} ({prediction_total/ground_truth_total:.2%})")

    actual = data['original_classification']
    predicted = data['greedy_prediction']
    
    if not superfamily:
        # Map classes to order level
        actual = actual.str.split("/").str[0]
        predicted = predicted.str.split("/").str[0]

    # Map values
    actual = np.array(map_replace_list(actual, map))
    predicted = np.array(map_replace_list(predicted, map))

    # count the correct predictions
    correct = (actual == predicted).sum()
    print(f"Correct predictions: {correct}/{prediction_total} ({correct/prediction_total:.2%})")
    
    labels = sorted(set(actual) | set(predicted))
    
    # Create a confusion matrix
    cm = sklearn_confusion_matrix(actual, predicted, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def confusion_matrix_fig(confusion_matrix:pd.DataFrame, width:int=800, height:int=800) -> go.Figure:
    # Normalize the confusion matrix (so long as there are no 0s in the denominator)
    sums = confusion_matrix.sum(axis=1).to_numpy()[:,None]
    zz = confusion_matrix.to_numpy()/np.maximum(sums, 1)
    
    z_text = [[str(y) for y in x] for x in confusion_matrix.to_numpy()]
    fig = px.imshow(zz, x=confusion_matrix.index, y=confusion_matrix.columns, color_continuous_scale='Viridis', aspect="equal", labels=dict(color="Sensitivity"))
    fig.update_traces(text=z_text, texttemplate="%{text}")

    format_fig(fig)
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
    )
    fig.update_xaxes(side="top")
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Ground Truth")
    return fig