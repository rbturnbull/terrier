from terrier.evaluate import evaluate_metrics, confusion_matrix_fig, build_confusion_matrix, format_fig
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path

def write_fig(fig, path:str|Path):
    path = str(path)
    fig.write_image(path+".png")
    fig.write_image(path+".svg")
    fig.write_html(path+".html")


datasets = ["Rice", "Fruit Fly"]
software_mapping_threshold = [
    ("DeepTE", "hAT-hobo=hAT,hAT-Tip100=hAT,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-EnSpm=CACTA", None),
    ("TERL", "/Pao=/Bel-Pao,DNA=TIR,CMC-Transib=CACTA,CMC-EnSpm=CACTA,TcMar-Tc1=Tc1-Mariner,hAT-hobo=hAT,TcMar-Pogo=Tc1-Mariner,hAT-Tip100=hAT", None),
    ("TEclass2", "I-Jockey=Jockey,/I=Jockey,TcMar-Tc1=TcMar,CMC-Transib=Transib,hAT-hobo=hAT,TcMar-Pogo=TcMar,hAT-Tip100=hAT,CMC-EnSpm=CACTA,/L1=/L1_L2", 0.7),
    ("TEclass2", "I-Jockey=Jockey,/I=Jockey,TcMar-Tc1=TcMar,CMC-Transib=Transib,hAT-hobo=hAT,TcMar-Pogo=TcMar,hAT-Tip100=hAT,CMC-EnSpm=CACTA,/L1=/L1_L2", 0.9),
    ("Terrier", "/I-Jockey=/I,/Jockey=/I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC", 0.7),
    ("Terrier", "/I-Jockey=/I,/Jockey=/I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC", 0.9),
]
data = []
base_dir = Path(__file__).parent
for software, mapping, threshold in software_mapping_threshold:
    row_index = software
    software_name_unique = software
    if threshold is not None:
        row_index += f" ({threshold})"
        software_name_unique += f"-{threshold}"
    row = dict(Software=row_index)

    software_dir = base_dir/software
    for dataset in datasets:
        dataset_clean = dataset.replace(" ", "-").lower()
        df = pd.read_csv(software_dir/f"{software}-{dataset_clean}.csv")
        confusion_matrix_dir = software_dir/f"{software}-{dataset_clean}-confusion-matrices"
        confusion_matrix_dir.mkdir(parents=True, exist_ok=True)
        row[f"{dataset} Order Proportion Classified"], row[f"{dataset} Order Accuracy"] = evaluate_metrics(df, map=mapping, superfamily=False, threshold=threshold)
        confusion = build_confusion_matrix(df, map=mapping, superfamily=False, threshold=threshold)
        fig = confusion_matrix_fig(confusion, title=f"{row_index} {dataset} Order Confusion Matrix", width=750, height=700)
        write_fig(fig, confusion_matrix_dir/f"{software_name_unique}-{dataset_clean}-order-confusion-matrix")

        row[f"{dataset} Superfamily Proportion Classified"], row[f"{dataset} Superfamily Accuracy"] = evaluate_metrics(df, map=mapping, superfamily=True, threshold=threshold)
        confusion = build_confusion_matrix(df, map=mapping, superfamily=True, threshold=threshold)
        fig = confusion_matrix_fig(confusion, title=f"{row_index} {dataset} Superfamily Confusion Matrix", width=750, height=700)
        write_fig(fig, confusion_matrix_dir/f"{software_name_unique}-{dataset_clean}-superfamily-confusion-matrix")
    data.append(row)

results = pd.DataFrame(data, columns=[
    'Software',
    'Rice Order Proportion Classified',
    'Rice Order Accuracy',
    'Rice Superfamily Proportion Classified',
    'Rice Superfamily Accuracy',
    'Fruit Fly Order Proportion Classified',
    'Fruit Fly Order Accuracy',
    'Fruit Fly Superfamily Proportion Classified',
    'Fruit Fly Superfamily Accuracy',
])

results.to_csv(base_dir/"comparison-results.csv")

for rank in ["Order", "Superfamily"]:
    fig = go.Figure()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("a. Rice", "b. Fruit Fly"), shared_yaxes=True, horizontal_spacing=0.03)
    fig.add_trace(
        go.Scatter(
            x=results[f"Rice {rank} Proportion Classified"],
            y=results[f"Rice {rank} Accuracy"],
            text = results["Software"],
            name=rank,
            showlegend=False,
            mode='markers+text',
            marker_color="blue",
            marker_size=12,
            marker_line_color="blue",
            textposition='bottom center',
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=results[f"Fruit Fly {rank} Proportion Classified"],
            y=results[f"Fruit Fly {rank} Accuracy"],
            text = results["Software"],
            name=rank,
            showlegend=False,
            mode='markers+text',
            marker_color="blue",
            marker_size=12,
            marker_line_color="blue",
            textposition='bottom center',
        ),
        row=1, col=2,
    )
    format_fig(fig)
    fig.update_layout(
        xaxis_title="Proportion Classified",
        yaxis_title=f"{rank} Accuracy",
        xaxis_tickformat=".0%",
        yaxis_tickformat=".0%",
        yaxis2_title="",  # Remove y-axis title for subplot 1
        xaxis2_tickformat=".0%",
        xaxis2_title="Proportion Classified",
        xaxis1_range=[0, 1.025],
        xaxis2_range=[0, 1.025],
        yaxis1_range=[0, 1.025],
        width=1300,
        height=700,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=10))
    write_fig(fig, base_dir/f"{rank}-Proportion-Classified-vs-Accuracy")

results.iloc[:, 1:] = results.iloc[:, 1:] * 100  # Multiply numeric columns by 100
results.iloc[:, 1:] = results.iloc[:, 1:].round(1).astype(str) + '\%'  # Format as percentage strings

# Export to LaTeX
latex_code = results.to_latex(index=False, escape=False)  # escape=False to handle '%' in LaTeX
print(latex_code)
with open(base_dir/"comparison-results.tex", "w") as f:
    f.write(latex_code)
