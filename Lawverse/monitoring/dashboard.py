import os
import json
import pandas as pd
from flask import Blueprint, render_template_string
from Lawverse.evaluation.ragas_eval import run_ragas_evaluation
from Lawverse.pipeline.llm_loader import llm

monitor_bp = Blueprint("monitor", __name__, url_prefix="/monitoring")

EVAL_DATA = [
    {
        "question": "What are the main objectives of the Digital Security Act, 2018?",
        "ground_truth": "The Digital Security Act, 2018 was enacted to ensure national digital security and to provide legal measures for the identification, prevention, suppression and trial of digital or cyber offences. It aims to protect citizens, public life, and property from digital crimes and to support implementation of 'Digital Bangladesh' under Vision 2021."
    },
    {
        "question": "What are the rights of workers under the Bangladesh Labour Act regarding working hours?",
        "ground_truth": "An adult worker shall ordinarily not be required to work more than eight hours in any day or forty-eight hours in a week. The Act mandates rest or meal intervals—no worker may work more than six hours without a one-hour break. The Government may set special limits for hazardous or laborious occupations."
    },
    {
        "question": "What is the process of company registration under the Companies Act, 1994?",
        "ground_truth": "Company registration is administered by the Registrar of Joint Stock Companies and Firms (RJSC). Steps include: obtaining name clearance; drafting the memorandum and articles of association; submitting incorporation forms and subscriber details; paying statutory fees; and receiving a certificate of incorporation after RJSC verification."
    },
    {
        "question": "সাইবার অপরাধের শাস্তি কী?",
        "ground_truth": "ডিজিটাল সিকিউরিটি অ্যাক্ট ২০১৮ অনুযায়ী সাইবার অপরাধের শাস্তি অপরাধের ধরন অনুযায়ী ভিন্ন। উদাহরণস্বরূপ, সোর্স কোড লুকানো বা ধ্বংস করার অপরাধে সর্বোচ্চ তিন বছর কারাদণ্ড বা তিন লাখ টাকা জরিমানা বা উভয় দণ্ড হতে পারে; পুনরাবৃত্তিতে শাস্তি বৃদ্ধি পায়। অন্যান্য অপরাধে পাঁচ থেকে সাত বছর পর্যন্ত কারাদণ্ড ও জরিমানা প্রযোজ্য।"
    },
    {
        "question": "শ্রমিকদের ছুটি সংক্রান্ত অধিকারগুলো কীভাবে নির্ধারিত হয়েছে?",
        "ground_truth": "বাংলাদেশ শ্রম আইন ২০০৬ অনুযায়ী শ্রমিকরা অর্জিত ছুটি, অসুস্থতা ছুটি, নৈমিত্তিক ছুটি ও উৎসব ছুটির অধিকার পায়। এক বছর ধারাবাহিক চাকরির পরে কর্মী অর্জিত ছুটি পান, এবং কোন উৎসব দিবসে কাজ করলে বদলি ছুটি বা অতিরিক্ত মজুরি দিতে হয়।"
    },
    {
        "question": "Under the Companies Act, 1994 what are the duties of directors of a company?",
        "ground_truth": "Directors must act in good faith for the benefit of the company, exercise due care and diligence, avoid conflicts of interest, maintain proper accounts, and comply with statutory filings. They are collectively responsible for ensuring that the company operates within the law and its memorandum and articles."
    },
    {
        "question": "According to the Bangladesh Labour Act, 2006 what safety and health obligations must an employer fulfil?",
        "ground_truth": "Employers must ensure workplace safety, adequate ventilation, lighting, cleanliness, safe disposal of waste, and protection of machinery. They must also provide medical facilities, first-aid arrangements, and measures against fire and occupational hazards as prescribed by the Act."
    },
    {
        "question": "ডিজিটাল সিকিউরিটি অ্যাক্টের আওতায় “ভয়-প্ররোচক তথ্য” প্রকাশের জন্য কী ধরণের দণ্ড রয়েছে?",
        "ground_truth": "যদি কেউ জেনে-বুঝে ডিজিটাল মাধ্যমে ভয়-প্ররোচক, মিথ্যা বা মানহানিকর তথ্য প্রচার করে, তবে প্রথমবার সর্বোচ্চ পাঁচ বছর কারাদণ্ড বা পাঁচ লাখ টাকা জরিমানা বা উভয় দণ্ড এবং পুনরাবৃত্তিতে সর্বোচ্চ সাত বছর কারাদণ্ড বা দশ লাখ টাকা জরিমানা প্রযোজ্য।"
    },
    {
        "question": "Under the Companies Act, 1994 what are the differences between a private company and a public company?",
        "ground_truth": "A private company restricts share transfer, limits members to fifty, and prohibits public subscription for shares, whereas a public company has no such restriction and may invite public investment. Only public companies may list shares on a stock exchange."
    },
    {
        "question": "শ্রমিকরা মায়ের ছুটি বা মাতৃত্বকালীন ছুটি বিষয়ে কি অধিকার পান?",
        "ground_truth": "বাংলাদেশ শ্রম আইন অনুযায়ী নারী শ্রমিকরা মোট ১৬ সপ্তাহ মাতৃত্বকালীন ছুটি পাওয়ার অধিকারী—প্রসবের আগে ৮ সপ্তাহ এবং পরে ৮ সপ্তাহ। ছুটিকালে স্বাভাবিক মজুরি প্রদান বাধ্যতামূলক, এবং নিয়োগকর্তা গর্ভাবস্থার কারণে চাকরি থেকে অব্যাহতি দিতে পারেন না।"
    },
    {
        "question": "What rights do shareholders have under the Companies Act, 1994 for winding up a company?",
        "ground_truth": "Shareholders may petition the court for winding up on grounds such as inability to pay debts, just and equitable causes, or failure of statutory meetings. They are entitled to receive proportional distribution of remaining assets after liabilities are settled."
    }
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