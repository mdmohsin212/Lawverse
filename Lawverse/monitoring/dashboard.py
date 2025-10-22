import os
import json
import pandas as pd
from flask import Blueprint, render_template_string
from Lawverse.evaluation.ragas_eval import run_ragas_evaluation
from Lawverse.pipeline.llm_loader import llm

monitor_bp = Blueprint("monitor", __name__, url_prefix="/monitoring")

EVAL_DATA = [
    {"question": "What is the punishment for cybercrime?", "ground_truth": "According to the Digital Security Act, 2018"},
    {"question": "Who can form a company in Bangladesh?", "ground_truth": "Under the Companies Act, 1994,"}
]

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG Monitoring Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial; margin:0; background:#f7f9fc; color:#222; }
        h1 { text-align:center; margin-top:20px; }
        #chart { width:90%%; margin:auto; }
        .latest { background:#fff; margin:2rem auto; padding:1rem 2rem; border-radius:12px; width:80%%; box-shadow:0 0 10px rgba(0,0,0,0.1); }
        pre { background:#f2f2f2; padding:1rem; border-radius:8px; overflow-x:auto; }
    </style>
</head>
<body>
    <h1>📊 Lawverse RAG Monitoring Dashboard</h1>
    <div id="chart"></div>
    <div class="latest">
        <h2>📋 Latest Metrics</h2>
        <pre id="latest"></pre>
    </div>

    <script>
        const data = {{ data_json | safe }};
        const latest = {{ latest_json | safe }};
        const traces = [];
        const metrics = [...new Set(data.map(d => d.metric))];
        metrics.forEach(metric => {
            const mdata = data.filter(d => d.metric === metric);
            traces.push({ x: mdata.map(d => d.timestamp), y: mdata.map(d => d.value), mode:'lines+markers', name:metric });
        });
        const layout = { title:'RAG Evaluation Metrics Over Time', xaxis:{title:'Timestamp'}, yaxis:{title:'Score (0-1)'} };
        Plotly.newPlot('chart', traces, layout);
        document.getElementById('latest').textContent = JSON.stringify(latest, null, 2);
    </script>
</body>
</html>
"""

@monitor_bp.route("/")
def dashboard():
    results = run_ragas_evaluation(EVAL_DATA, llm)
    METRICS_JSON_PATH = "monitoring/rag_metrics.json"
    if os.path.exists(METRICS_JSON_PATH):
        with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
            logs = json.load(f)
    else:
        logs = []
        
    records = []
    for entry in logs:
        ts = entry["timestamp"]
        for metric, value in entry["metrics"].items():
            records.append({"timestamp": ts, "metric": metric, "value": value})
    df = pd.DataFrame(records)

    latest = logs[-1] if logs else {}
    return render_template_string(TEMPLATE, data_json=df.to_json(orient="records"), latest_json=json.dumps(latest, indent=2))