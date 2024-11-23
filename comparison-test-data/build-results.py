from terrier.evaluate import evaluate_metrics, confusion_matrix_fig, build_confusion_matrix
import pandas as pd
from pathlib import Path

def write_fig(fig, path:str|Path):
    path = str(path)
    fig.write_image(path+".png")
    fig.write_image(path+".pdf")
    fig.write_html(path+".html")


datasets = ["Rice", "Fruit Fly"]
software_mapping_threshold = [
    ("DeepTE", "hAT-hobo=hAT,hAT-Tip100=hAT,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-EnSpm=CACTA", None),
    ("TERL", "/Pao=/Bel-Pao,DNA=TIR,CMC-Transib=CACTA,CMC-EnSpm=CACTA,TcMar-Tc1=Tc1-Mariner,hAT-hobo=hAT,TcMar-Pogo=Tc1-Mariner,hAT-Tip100=hAT", None),
    ("TEclass2", "I-Jockey=Jockey,/I=Jockey,TcMar-Tc1=TcMar,CMC-Transib=Transib,hAT-hobo=hAT,TcMar-Pogo=TcMar,hAT-Tip100=hAT,CMC-EnSpm=CACTA,/L1=/L1_L2", 0.7),
    ("TEclass2", "I-Jockey=Jockey,/I=Jockey,TcMar-Tc1=TcMar,CMC-Transib=Transib,hAT-hobo=hAT,TcMar-Pogo=TcMar,hAT-Tip100=hAT,CMC-EnSpm=CACTA,/L1=/L1_L2", 0.9),
    ("Terrier", "I-Jockey=I,Jockey-I=I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC", 0.7),
    ("Terrier", "I-Jockey=I,Jockey-I=I,TcMar-Pogo=TcMar,TcMar-Tc1=TcMar,CMC-Transib=CMC,R1-LOA=R1,hAT-hobo=hAT,hAT-Tip100=hAT,CMC-EnSpm=CMC", 0.9),
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

results.iloc[:, 1:] = results.iloc[:, 1:] * 100  # Multiply numeric columns by 100
results.iloc[:, 1:] = results.iloc[:, 1:].round(1).astype(str) + '\%'  # Format as percentage strings

# Export to LaTeX
latex_code = results.to_latex(index=False, escape=False)  # escape=False to handle '%' in LaTeX
print(latex_code)
