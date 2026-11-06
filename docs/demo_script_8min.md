# 8-minute maximum demonstration video script

Keep the screen recording to **8:00 maximum**. Replace all bracketed text with actual values from the completed real-data run. Do not show the app with no trained artifacts and pretend a prediction succeeded.

| Time | Screen / action | Narration |
|---|---|---|
| 0:00–0:35 | Title and problem statement | “This project forecasts next-hour benzene concentration using a historical sequence of low-cost sensor and weather observations. The intended user is an environmental monitoring station operator, and the decision is whether to prioritise an internal review or station inspection.” |
| 0:35–1:15 | Dataset source and audit | Show UCI dataset citation, 9,358 source records, time coverage, target and missingness audit. Explain `-200` is missing. Mention the UCI research/non-commercial restriction. |
| 1:15–2:00 | EDA figures | Show target series, missingness, target distribution and correlation heatmap. State three measured findings and the modelling consequence of each. |
| 2:00–2:45 | Leakage/split and ANN architecture | Show `leakage_audit.json`, the 70/15/15 chronological split, 24-hour input window and one-hour horizon. Explain why ground-truth pollutant channels are excluded. Show LSTM design. |
| 2:45–3:40 | Training, baseline and experiments | Show experiment log, four unsuccessful/not-selected attempts, two optimisers and baseline search. Read actual validation and held-out test MAE/RMSE/R². Do not conceal if baseline wins. |
| 3:40–4:35 | Web application live | Upload a filled 24-hour sensor input CSV. Generate a request-time forecast. Explain the validation MAE is not a confidence interval and state the app limitations. |
| 4:35–5:20 | Explainability | Show Integrated Gradients explanation and the two perturbation checks. State that attribution is not causal. |
| 5:20–6:10 | Error analysis | Show predicted-vs-actual, residual plot and largest error. Explain at least one meaningful error pattern from actual results. |
| 6:10–7:00 | Generative model | If lecturer-approved and API configured, show a generated handover note and exact model name/latency/tokens. Explain it does not change the forecast. Otherwise show the fixed prompt and explain why the optional call was not executed. |
| 7:00–7:40 | Reproducibility / limitations | Show project structure, requirements, downloader, experiment log and report. Explain single-site historical-data limitation and commercial-use caution. |
| 7:40–8:00 | Closing | State the contribution and the next validation step with current local measurements and a certified reference analyser. |

## Recording checklist

- The ZIP contains the real UCI CSV and `dataset_manifest.json` after download.
- `models/` has trained artifacts created by the actual dataset pipeline.
- The prediction displayed is from `streamlit_app.py` at request time.
- The app input CSV uses only sensor/weather fields, not target or other reference-analyser concentrations.
- Figures, scores and experiment log match the final report.
- No API key or secret appears on screen.
- Total recording length is no more than 8:00.
