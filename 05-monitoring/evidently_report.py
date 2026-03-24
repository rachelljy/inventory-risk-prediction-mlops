# evidently_report.py

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

df = pd.read_csv("predictions.csv")

reference = df.iloc[:50]
current = df.iloc[50:]

report = Report([
    DataDriftPreset(),
])

my_eval = report.run(current_data=current, reference_data=reference)
my_eval.save_html("drift_report.html")

print("Drift report saved as drift_report.html")