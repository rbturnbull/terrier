from terrier.evaluate import evaluate_metrics, confusion_matrix_fig, build_confusion_matrix, format_fig
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path

def write_fig(fig, path:str|Path):
    path = str(path)
    fig.write_image(path+".png")
    fig.write_image(path+".pdf")
    fig.write_html(path+".html")

IGNORE = ["Unknown", "ARTEFACT"]

BASE_RULES  = "/Pao=/Bel-Pao,TIR=DNA,DNA/CMC-.*=DNA/CACTA,DNA/CMC=DNA/CACTA,TcMar-.*=Tc1,Tc1-.*=Tc1,hAT-.*=hAT,LTR/ERV.*=LTR/ERV,L1-.*=L1,PIF-Harbinger=Harbinger,Crypton-.*=Crypton,RTE-.*=RTE,Retroposon/L1=LINE/L1,Satellite/.*=Satellite,^tRNA=SINE/tRNA,SINE/tRNA-.*=SINE/tRNA,TcMar=Tc1,SINE/5S-.*=SINE/5S,SINE/Alu=SINE/7SL,SINE/B2=SINE/tRNA,SINE/B4=SINE/tRNA,SINE/MIR=SINE/tRNA,SINE/ID=SINE/tRNA,/I-Jockey=/I,/Jockey.*=/I,/MULE-.*=/MULE,LINE/R1-.*=LINE/R1"
DEEPTE_RULES = BASE_RULES + ",_nMITE=,_MITE="
TERL_RULES = BASE_RULES
TECLASS2_RULES = BASE_RULES + ",LINE/L1$=LINE/L1_L2,LINE/L2$=LINE/L1_L2"
TERRIER_RULES = BASE_RULES

datasets = ["Rice", "Fruit Fly", "Human", "Mouse"]
software_mapping_threshold = [
    ("DeepTE", DEEPTE_RULES, None),
    ("TERL", TERL_RULES, None),
    ("TEclass2", TECLASS2_RULES, 0.7),
    ("TEclass2", TECLASS2_RULES, 0.9),
    ("Terrier", TERRIER_RULES, 0.7),
    ("Terrier", TERRIER_RULES, 0.9),
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
        row[f"{dataset} Order Proportion Classified"], row[f"{dataset} Order Accuracy"] = evaluate_metrics(df, map=mapping, superfamily=False, threshold=threshold, ignore=IGNORE)
        confusion = build_confusion_matrix(df, map=mapping, superfamily=False, threshold=threshold)
        fig = confusion_matrix_fig(confusion, title=f"{row_index} {dataset} Order Confusion Matrix", width=750, height=700)
        write_fig(fig, confusion_matrix_dir/f"{software_name_unique}-{dataset_clean}-order-confusion-matrix")

        row[f"{dataset} Superfamily Proportion Classified"], row[f"{dataset} Superfamily Accuracy"] = evaluate_metrics(df, map=mapping, superfamily=True, threshold=threshold, ignore=IGNORE + ["DNA", "LTR"])
        confusion = build_confusion_matrix(df, map=mapping, superfamily=True, threshold=threshold)
        fig = confusion_matrix_fig(confusion, title=f"{row_index} {dataset} Superfamily Confusion Matrix", width=750, height=700)
        write_fig(fig, confusion_matrix_dir/f"{software_name_unique}-{dataset_clean}-superfamily-confusion-matrix")
    data.append(row)

columns = ["Software"]
for dataset in datasets:
    columns.append(f"{dataset} Order Proportion Classified")
    columns.append(f"{dataset} Order Accuracy")
    columns.append(f"{dataset} Superfamily Proportion Classified")
    columns.append(f"{dataset} Superfamily Accuracy")

results = pd.DataFrame(data, columns=columns)

results.to_csv(base_dir/"comparison-results.csv")

subplot_titles = [f"{chr(97 + i)} {dataset}" for i, dataset in enumerate(datasets)]

for rank in ["Order", "Superfamily"]:
    fig = go.Figure()
    fig = make_subplots(rows=2, cols=2, subplot_titles=subplot_titles, shared_yaxes=True, horizontal_spacing=0.03, vertical_spacing=0.06)
    software_labels = [label.replace(" ","<br>") for label in results["Software"]]
    for dataset_index, dataset in enumerate(datasets):
        fig.add_trace(
            go.Scatter(
                x=results[f"{dataset} {rank} Proportion Classified"],
                y=results[f"{dataset} {rank} Accuracy"],
                text = software_labels,
                name=rank,
                showlegend=False,
                mode='markers+text',
                marker_color="blue",
                marker_size=12,
                marker_line_color="blue",
                textposition='bottom center',
            ),
            row=1 + dataset_index // 2, col=1 + dataset_index % 2,
        )
    format_fig(fig)
    fig.update_layout(
        xaxis_title="",
        xaxis2_title="",
        xaxis3_title="Proportion Classified",
        xaxis4_title="Proportion Classified",
        yaxis_title=f"{rank} Accuracy",
        yaxis2_title=f"",
        yaxis3_title=f"{rank} Accuracy",
        yaxis4_title=f"",
        yaxis_tickformat=".0%",

        xaxis_tickformat=".0%",        
        xaxis2_tickformat=".0%",
        xaxis3_tickformat=".0%",
        xaxis4_tickformat=".0%",
        xaxis1_range=[0, 1.025],
        xaxis2_range=[0, 1.025],
        xaxis3_range=[0, 1.025],
        xaxis4_range=[0, 1.025],
        yaxis1_range=[0, 1.025],
        yaxis3_range=[0, 1.025],
        width=1300,
        height=1300,
    )
    fig.update_layout(margin=dict(l=0, r=0, t=20, b=10))
    write_fig(fig, base_dir/f"{rank}-Proportion-Accuracy-vs-Classified")

results.iloc[:, 1:] = results.iloc[:, 1:] * 100  # Multiply numeric columns by 100
results.iloc[:, 1:] = results.iloc[:, 1:].round(1).astype(str) + '\%'  # Format as percentage strings

# Export to LaTeX
latex_code1 = results[results.columns[:9]].to_latex(index=False, escape=False)  # escape=False to handle '%' in LaTeX
latex_code2 = results[results.columns[0:1].to_list() + results.columns[9:].to_list()].to_latex(index=False, escape=False)
print(latex_code1)
print(latex_code2)
with open(base_dir/"comparison-results.tex", "w") as f:
    f.write(latex_code1)
    f.write("\n\n")
    f.write(latex_code2)
