from pathlib import Path
import pandas as pd
import re
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import plotly.io as pio   
pio.kaleido.scope.mathjax = None

DEFAULT_WIDTH = 730
DEFAULT_HEIGHT = 690


def build_map(map:str) -> dict[str,str]:
    """
    Builds a dictionary from a string in the form of 'key1=value1|key2=value2'
    """
    if not map:
        return {}
    components = map.split(",")
    return {x.split("=")[0]: x.split("=")[1] for x in components}


def build_str_list(string:str) -> list[str]:
    """
    Builds a dictionary from a string in the form of 'key1=value1|key2=value2'
    """
    if not string:
        return []
    return [x.strip() for x in string.split(",")]


def map_replace(string:str, map:dict[str,str]) -> str:
    """
    Replaces values in a string using a map.
    """
    for key, value in map.items():
        string = re.sub(key, value, string)
        # string = string.replace(key, value)
    return string


def map_replace_list(iterable:list[str], map:dict[str,str]) -> list[str]:
    """
    Replaces values in an iterable using a map.
    """
    return [map_replace(x, map) for x in iterable]


def format_fig(fig):
    """Formats a plotly figure in a nicer way."""
    fig.update_layout(
        width=DEFAULT_WIDTH,
        height=DEFAULT_HEIGHT,
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


def evaluate_results(data:pd.DataFrame, superfamily:bool=True, map:str|dict="", ignore:str|list[str]="Unknown", threshold:float|None=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(map, str):
        map = build_map(map)
    
    if isinstance(ignore, str):
        ignore = build_str_list(ignore)

    total = len(data)
    print(f"Total: {total}")

    # Remove ignored classes
    data = data.fillna('')
    data = data[~data['original_classification'].isin(ignore)].copy()

    # Filter ground truth
    ground_truth_total = len(data)
    print(f"Total with ground truth: {ground_truth_total}")

    # Map classes to order level
    if not superfamily:
        exclude_columns = ['file', 'accession', 'prediction', 'probability', 'original_id', 'original_classification']
        filtered_columns = [col for col in data.columns if col not in exclude_columns and '/' not in col]
        if filtered_columns:
            data["prediction"] = data[filtered_columns].idxmax(axis=1)
            data["probability"] = data[filtered_columns].max(axis=1)
        else:
            data["prediction"] = data["prediction"].str.split("/").str[0]

        data["original_classification"] = data["original_classification"].str.split("/").str[0]
    else:
        data["prediction"] = data["prediction"].apply(lambda x: x if '/' in x else "")

    # Filter predictions
    if threshold is not None:
        data = data[data['probability'] >= threshold]
    
    # Ignore empty predictions
    data = data[data['prediction'] != ""]

    prediction_total = len(data)
    print(f"Number classified: {prediction_total}/{ground_truth_total} ({prediction_total/ground_truth_total:.2%})")

    actual = data['original_classification']
    predicted = data['prediction']
    probability = data['probability'].to_numpy() if 'probability' in data.columns else np.ones(len(data))
    
    # Map values
    actual = np.array(map_replace_list(actual, map))
    predicted = np.array(predicted)

    # count the correct predictions
    correct = (actual == predicted).sum()
    print(f"Correct predictions: {correct}/{prediction_total} ({correct/prediction_total:.2%})")

    return actual, predicted, probability


def evaluate_metrics(data:pd.DataFrame, superfamily:bool=True, map:str|dict="", ignore:str|list[str]="Unknown", threshold:float|None=None) -> tuple[float, float]:
    actual, predicted, _ = evaluate_results(data, map=map, superfamily=superfamily, ignore=ignore, threshold=threshold)
    classified_proportion = len(predicted)/len(data)
    accuracy = (actual == predicted).mean()

    return classified_proportion, accuracy


def threshold_fig(data:pd.DataFrame, superfamily:bool=True, map:str|dict="", ignore:str|list[str]="Unknown", width:int=DEFAULT_WIDTH, height:int=DEFAULT_HEIGHT) -> go.Figure:
    actual, predicted, probability = evaluate_results(data, superfamily, map, ignore, threshold=None)
    df = pd.DataFrame({"actual": actual, "predicted": predicted, "probability": probability})
    df = df.sort_values("probability", ascending=False)

    df["classified"] = (np.arange(len(df))+1)/len(df)
    df["correct"] = (df["actual"] == df["predicted"]).cumsum()/(np.arange(len(df))+1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["probability"], y=df["correct"], mode='lines', name='Correct'))
    fig.add_trace(go.Scatter(x=df["probability"], y=df["classified"], mode='lines', name='Classified'))
    format_fig(fig)
    fig.update_xaxes(title="Threshold")
    fig.update_yaxes(title="Percentage of dataset", tickformat=".0%", range=[0, 1])
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
        ),
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

    return fig


def build_confusion_matrix(data:pd.DataFrame, superfamily:bool=True, map:str|dict="", ignore:str|list[str]="Unknown", threshold:float|None=None) -> pd.DataFrame:
    actual, predicted, _ = evaluate_results(data, superfamily, map, ignore, threshold)
    labels = sorted(set(actual) | set(predicted))
    
    # Create a confusion matrix
    cm = sklearn_confusion_matrix(actual, predicted, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)


def confusion_matrix_fig(confusion_matrix:pd.DataFrame, width:int=DEFAULT_WIDTH, height:int=DEFAULT_HEIGHT, title:str="Confusion Matrix") -> go.Figure:
    # Normalize the confusion matrix (so long as there are no 0s in the denominator)
    sums = confusion_matrix.sum(axis=1).to_numpy()[:,None]
    zz = confusion_matrix.to_numpy()/np.maximum(sums, 1)
    
    z_text = [[str(y) for y in x] for x in confusion_matrix.to_numpy()]
    fig = px.imshow(zz, x=confusion_matrix.index, y=confusion_matrix.columns, color_continuous_scale='Viridis', aspect="equal", labels=dict(color="Class<br>Percent"))
    fig.update_traces(text=z_text, texttemplate="%{text}")

    format_fig(fig)
    fig.update_layout(
        coloraxis_colorbar=dict(
            len=1.05,
            x=1,
            tickformat=".0%",
        )
    )
    # Set margins to zero except for bottom
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=40))
    fig.update_xaxes(side="top")
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Ground Truth")
    
    fig.update_layout(
        autosize=False,
        title={
            'text': title,
            'y': 0.01,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'bottom',
        },
        width=width,
        height=height,
        xaxis=dict(scaleanchor=None),
        yaxis=dict(scaleanchor=None),
    )
    return fig


def comparison_plot(files: list[Path], superfamily:bool=False, threshold:float=0.0):
    def read_csv(csv_file: Path):
        df = pd.read_csv(csv_file)
        df["original_classification"] = df["original_classification"].str.replace("?","")
        return df

    height = 400+300*len(files)
    fig = make_subplots(
        rows=len(files), cols=2, 
        shared_yaxes=True, shared_xaxes=True, 
        horizontal_spacing=0.03,
        vertical_spacing=20 / height, 
        subplot_titles=("Original classification", "Terrier"),
    )

    categories = set()
    for index, csv_file in enumerate(files):
        df = read_csv(csv_file)

        if threshold > 0.0:
            indexes = df["probability"] > threshold
            df.loc[ indexes, "prediction" ] = "Unknown"

        if not superfamily:
            df["original_classification"] = df["original_classification"].str.split("/").str[0]
            df["prediction"] = df["prediction"].str.split("/").str[0]

        categories |= set(df["original_classification"].unique()) | set(df["prediction"].unique())

    categories = sorted(categories, key=lambda x: x.lower())

    for index, csv_file in enumerate(files):
        df = read_csv(csv_file)

        if threshold > 0.0:
            indexes = df["probability"] > threshold
            df.loc[ indexes, "prediction" ] = "Unknown"

        if not superfamily:
            df["original_classification"] = df["original_classification"].str.split("/").str[0]
            df["prediction"] = df["prediction"].str.split("/").str[0]

        original_classified = df.loc[df["original_classification"] != "Unknown", "original_classification"].value_counts()
        original_classified = [original_classified[k] if k in original_classified else 0 for k in categories]

        original_unclassified = df.loc[df["original_classification"] == "Unknown", "original_classification"].value_counts()
        original_unclassified = [original_unclassified[k] if k in original_unclassified else 0 for k in categories]

        # agreement = df.loc[(df["prediction"] == df["original_classification"]), "prediction"].value_counts()
        # disagreement = df.loc[(df["prediction"] != df["original_classification"]) & (df["original_classification"] != "Unknown"), "prediction"].value_counts()
        # unknown = df.loc[(df["prediction"] != df["original_classification"]) & (df["original_classification"] == "Unknown"), "prediction"].value_counts()

        agreement = df.loc[(df["prediction"] == df["original_classification"]) & (df["original_classification"] != "Unknown"), "prediction"].value_counts()
        disagreement = df.loc[(df["prediction"] != df["original_classification"]) & (df["original_classification"] != "Unknown"), "prediction"].value_counts()
        unknown = df.loc[(df["original_classification"] == "Unknown"), "prediction"].value_counts()

        agreement = [agreement[k] if k in agreement else 0 for k in categories]
        disagreement = [disagreement[k] if k in disagreement else 0 for k in categories]
        unknown = [unknown[k] if k in unknown else 0 for k in categories]

        fig.add_trace(go.Bar(x=categories, y=disagreement, name="Terrier Disagree", showlegend=index==0, marker_color="red"), row=1+index, col=2)
        fig.add_trace(go.Bar(x=categories, y=agreement, name="Terrier Agree", showlegend=index==0, marker_color="blue"), row=1+index, col=2)

        fig.add_trace(go.Bar(x=categories, y=original_unclassified, name="Original Unclassified", showlegend=index==0, marker_color="grey"), row=1+index, col=1)
        fig.add_trace(go.Bar(x=categories, y=original_classified, name="Original Classified", showlegend=index==0, marker_color="orange"), row=1+index, col=1)

        fig.add_trace(go.Bar(x=categories, y=unknown, name="Original Unclassified", showlegend=False, marker_color="grey"), row=1+index, col=2)

        filename = csv_file.stem.replace("-families-terrier", "")
        filename = filename.replace("_", " ")
        filename = filename[0].upper() + filename[1:]


        fig.update_layout(**{f"yaxis{1+index*2}_title": f"{filename}"})

    # stack barmode
    fig.update_layout(barmode='stack')

    # legend horizontal and on top
    legend_height_px = 30
    legend_y_position = 1 + (legend_height_px / height)

    fig.update_layout(
        margin=dict(t=legend_height_px + 20),  # Ensure space for the legend
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=legend_y_position,
            xanchor="right",
            x=1
        )
    )

    format_fig(fig)
    fig.update_layout(height=height, width=1000)
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=20))    

    return fig