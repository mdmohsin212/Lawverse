import json
from pathlib import Path
import pandas as pd
from flask import Blueprint, render_template_string

monitor_bp = Blueprint("monitor", __name__, url_prefix="/monitoring")

EVAL_DIR = Path("artifacts/evaluation")
REPORT_FILES = {
    "Retrieval": EVAL_DIR / "retrieval_metrics.json",
    "RAG Generation": EVAL_DIR / "rag_generation_metrics.json",
    "Agent Behavior": EVAL_DIR / "agent_metrics.json",
    "Safety": EVAL_DIR / "safety_metrics.json",
}

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Lawverse Evaluation Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin:0; background:#f7f9fc; color:#222; }
        h1 { text-align:center; margin-top:24px; }
        .note { width:86%%; margin:1rem auto; background:#fff8e6; border:1px solid #f0d58a; padding:1rem; border-radius:10px; }
        #scoreChart, #latencyChart { width:90%%; margin:1rem auto; }
        .latest { background:#fff; margin:2rem auto; padding:1rem 2rem; border-radius:12px; width:86%%; box-shadow:0 0 10px rgba(0,0,0,0.08); }
        pre { background:#f2f2f2; padding:1rem; border-radius:8px; overflow-x:auto; }
    </style>
</head>
<body>
    <h1>📊 Lawverse Evaluation Dashboard</h1>
    <div class="note text-center">
        This dashboard only reads saved evaluation reports from <code>artifacts/evaluation/</code>.
        It does not run LLM/RAG evaluation automatically.
    </div>
    <div id="scoreChart"></div>
    <div id="latencyChart"></div>
    <div class="latest">
        <h2>Latest Reports</h2>
        <pre id="latest"></pre>
    </div>

    <script>
    const data = {{ data_json | safe }};
    const latest = {{ latest_json | safe }};

    function isLatency(metric) {
        metric = metric.toLowerCase();
        return metric.includes("latency") || metric.endsWith("_ms");
    }

    const scoreData = data.filter(d => !isLatency(d.metric));
    const latencyData = data.filter(d => isLatency(d.metric));

    function drawChart(divId, chartData, title, yTitle, yRange=null) {
        const traces = [];
        const groups = [...new Set(chartData.map(d => d.report))];

        groups.forEach(group => {
            const mdata = chartData.filter(d => d.report === group);
            traces.push({
                x: mdata.map(d => d.metric),
                y: mdata.map(d => d.value),
                type: "bar",
                name: group
            });
        });

        const layout = {
            title: title,
            barmode: "group",
            yaxis: { title: yTitle, range: yRange },
            xaxis: { tickangle: -35 },
            margin: { b: 140 }
        };

        Plotly.newPlot(divId, traces, layout);
    }

    drawChart("scoreChart", scoreData, "Evaluation Score Metrics", "Score", [0, 1.05]);
    drawChart("latencyChart", latencyData, "Latency Metrics", "Milliseconds");

    document.getElementById("latest").textContent = JSON.stringify(latest, null, 2);
</script>
</body>
</html>
"""

def _read_json(path: Path):
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@monitor_bp.route("/")
def dashboard():
    reports = {name: _read_json(path) for name, path in REPORT_FILES.items()}
    records = []
    for report_name, report in reports.items():
        metrics = report.get("metrics", {}) if isinstance(report, dict) else {}
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                records.append({"report": report_name, "metric": metric, "value": value})

    df = pd.DataFrame(records)
    data_json = df.to_json(orient="records") if not df.empty else "[]"
    return render_template_string(TEMPLATE, data_json=data_json, latest_json=json.dumps(reports, indent=2))