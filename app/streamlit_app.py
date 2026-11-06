from __future__ import annotations
import io, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)
sys.path.insert(0, str(ROOT / "src"))
from airquality.config import BASE_FEATURES, TARGET, TARGET_UNIT
from airquality.inference import load_ann_artifacts, predict_from_sensor_history, check_training_range
from airquality.xai import integrated_gradients

st.set_page_config(page_title="Air Quality Intelligence", page_icon="🌫️", layout="wide")
st.title("Air Quality Intelligence")
st.caption("Research prototype · one-hour-ahead benzene forecasting · LSTM ANN")
st.warning("This prototype uses historical data from one Italian roadside station (2004–2005). It is not a regulatory monitoring instrument, and predictions must be confirmed with certified measurements before operational action.")


def parse_upload(file) -> pd.DataFrame:
    df = pd.read_csv(file, sep=None, engine="python", decimal=",")
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    df = df.dropna(axis=0, how="all")
    columns_by_casefold = {column.casefold(): column for column in df.columns}
    timestamp_column = columns_by_casefold.get("timestamp")
    date_column = columns_by_casefold.get("date")
    time_column = columns_by_casefold.get("time")
    if timestamp_column:
        index = pd.to_datetime(df.pop(timestamp_column), errors="coerce")
    elif date_column and time_column:
        date_text = df.pop(date_column).astype(str).str.strip()
        time_text = df.pop(time_column).astype(str).str.strip().str.replace(".", ":", regex=False)
        index = pd.to_datetime(date_text + " " + time_text, format="%d/%m/%Y %H:%M:%S", errors="coerce")
    else:
        raise ValueError("CSV needs a timestamp column or Date and Time columns. Comma- and semicolon-delimited CSV files are supported.")
    if index.isna().any():
        raise ValueError("Some timestamps could not be parsed. Use YYYY-MM-DD HH:MM:SS in timestamp, or DD/MM/YYYY and HH.MM.SS.")
    df.index = pd.DatetimeIndex(index, name="timestamp")
    missing = [c for c in BASE_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing sensor/weather columns: {missing}")
    display_columns = BASE_FEATURES + ([TARGET] if TARGET in df.columns else [])
    df = df[display_columns].apply(pd.to_numeric, errors="coerce").replace(-200, np.nan)
    return df.sort_index()


with st.sidebar:
    st.header("Model information")
    st.write("**Target:** Benzene (`C6H6(GT)`) ")
    st.write("**Horizon:** Next hour")
    st.write("**Inputs:** 24 hourly records from five gas sensors and three weather measurements")
    st.write("**Primary metric:** Validation MAE")
    st.download_button("Download input CSV template", data=(ROOT / "data" / "demo" / "input_template.csv").read_bytes() if (ROOT / "data" / "demo" / "input_template.csv").exists() else "timestamp," + ",".join(BASE_FEATURES), file_name="sensor_history_template.csv", mime="text/csv")

try:
    model, preprocessor, target_scaler, metadata = load_ann_artifacts(ROOT)
except Exception as exc:
    st.error("The trained model has not been created yet.")
    st.code("python scripts/download_data.py\npython scripts/run_pipeline.py\nstreamlit run app/streamlit_app.py", language="bash")
    st.caption(str(exc))
    st.stop()

forecast_tab, diagnostics_tab, profile_tab = st.tabs(["Forecast", "Model diagnostics", "Data profile"])

with forecast_tab:
    st.subheader("Submit recent sensor readings")
    st.write("Upload at least 24 consecutive hourly rows. Sensor/weather columns are used for prediction. Optionally include `C6H6(GT)` to show historical actuals; it is chart-only and never used as a model input.")
    upload = st.file_uploader("Upload sensor-history CSV", type=["csv"])
    if upload is not None:
        try:
            history = parse_upload(upload)
            if len(history) < metadata["window_size"]:
                st.error(f"At least {metadata['window_size']} rows are required; received {len(history)}.")
            else:
                recent = history.tail(metadata["window_size"])
                st.dataframe(recent, use_container_width=True)
                missing_count = int(recent.isna().sum().sum())
                if missing_count:
                    st.info(f"{missing_count} missing sensor/weather values will be imputed using medians fitted on the training period. Missingness indicators are included in the model.")
                range_check = check_training_range(recent, preprocessor)
                if range_check["outside_training_range_count"]:
                    st.warning(f"{range_check['outside_training_range_count']} readings fall outside the training-period 1st–99th percentile ranges. This is a potential distribution-shift warning; verify sensor calibration and confirm with a reference analyser.")
                else:
                    st.success("All observed sensor/weather values fall within their training-period 1st–99th percentile ranges.")

                chart_col, quality_col = st.columns([2, 1])
                with chart_col:
                    selected_feature = st.selectbox("Sensor trend", BASE_FEATURES, format_func=lambda value: value)
                    trend = px.line(recent.reset_index(), x="timestamp", y=selected_feature, markers=True,
                                    title=f"Latest 24-hour trend · {selected_feature}",
                                    labels={"timestamp": "Hour", selected_feature: selected_feature},
                                    template="plotly_white")
                    trend.update_traces(line_color="#087E8B", marker_color="#FF5A5F")
                    trend.update_layout(margin=dict(l=12, r=12, t=48, b=12), height=340)
                    st.plotly_chart(trend, use_container_width=True)
                with quality_col:
                    missing_by_feature = recent.isna().sum().rename_axis("feature").reset_index(name="missing_rows")
                    missing_fig = px.bar(missing_by_feature, x="missing_rows", y="feature", orientation="h",
                                         title="Missing readings", labels={"missing_rows": "Rows", "feature": ""},
                                         template="plotly_white", color="missing_rows",
                                         color_continuous_scale=["#DCE7E8", "#FF5A5F"])
                    missing_fig.update_layout(margin=dict(l=8, r=8, t=48, b=8), height=340,
                                              coloraxis_showscale=False)
                    st.plotly_chart(missing_fig, use_container_width=True)

                if st.button("Generate next-hour forecast", type="primary"):
                    prediction, model_input = predict_from_sensor_history(history, model, preprocessor, target_scaler, metadata)
                    st.session_state["last_prediction"] = prediction
                    st.session_state["last_model_input"] = model_input
                    st.session_state["last_history"] = recent
                    st.session_state["last_timestamp"] = str(recent.index[-1])
        except Exception as exc:
            st.error(str(exc))

    if "last_prediction" in st.session_state:
        prediction = st.session_state["last_prediction"]
        if prediction < 0:
            st.error(f"The unconstrained regression model returned {prediction:.2f} {TARGET_UNIT}, which is not a physically valid concentration. Do not use this output for an operational decision; check the sensor readings and confirm with a reference analyser.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Forecast benzene concentration", f"{prediction:.2f} {TARGET_UNIT}")
            c2.metric("Forecast horizon", "1 hour")
            c3.metric("Validation MAE", f"{metadata['validation_ann']['MAE']:.2f} {TARGET_UNIT}")
            st.caption(f"Forecast generated using readings through {st.session_state['last_timestamp']}. Validation MAE is not a prediction-specific confidence interval.")

        st.subheader("Historical actuals and next-hour forecast")
        forecast_history = st.session_state["last_history"]
        future_time = pd.Timestamp(st.session_state["last_timestamp"]) + pd.Timedelta(hours=1)
        forecast_fig = go.Figure()
        if TARGET in forecast_history.columns and forecast_history[TARGET].notna().any():
            actuals = forecast_history[TARGET].dropna()
            forecast_fig.add_trace(go.Scatter(
                x=actuals.index, y=actuals, mode="lines+markers", name="Past actual",
                line=dict(color="#087E8B", width=2), marker=dict(size=6),
                connectgaps=False,
            ))
            forecast_fig.add_trace(go.Scatter(
                x=[actuals.index[-1], future_time], y=[actuals.iloc[-1], prediction],
                mode="lines+markers", name="Forecast continuation",
                line=dict(color="#FF5A5F", width=2, dash="dash"),
                marker=dict(size=9, symbol="circle"),
            ))
        else:
            st.info(f"Historical `{TARGET}` values were not included in the upload, so only the next-hour forecast is shown. Add that optional column to plot past actuals; it will remain excluded from prediction inputs.")
        forecast_fig.add_trace(go.Scatter(
            x=[future_time], y=[prediction], mode="markers+text", name="Next-hour forecast",
            marker=dict(color="#FF5A5F", size=13, symbol="diamond"),
            text=[f"{prediction:.2f} {TARGET_UNIT}"], textposition="top center",
        ))
        forecast_fig.update_layout(
            title="Benzene concentration history and forecast",
            xaxis_title="Time", yaxis_title=f"Benzene ({TARGET_UNIT})",
            template="plotly_white", hovermode="x unified",
            margin=dict(l=12, r=12, t=52, b=12), height=420,
            legend_title_text="",
        )
        st.plotly_chart(forecast_fig, use_container_width=True)
        st.caption("The dashed segment connects the latest available actual to the forecast for visual continuity; it is not an additional model output. Historical actuals are optional and are not passed into the model.")

        st.subheader("Explain this forecast")
        try:
            attrs = integrated_gradients(model, st.session_state["last_model_input"][0], metadata["input_features"], steps=48, device="cpu").head(10)
            attribution_fig = px.bar(attrs.sort_values("absolute_attribution"), x="absolute_attribution", y="feature", orientation="h",
                                     title="Largest input contributions · Integrated Gradients",
                                     labels={"absolute_attribution": "Absolute attribution", "feature": "Input feature"},
                                     template="plotly_white", color="absolute_attribution",
                                     color_continuous_scale=["#087E8B", "#FF5A5F"])
            attribution_fig.update_layout(margin=dict(l=12, r=12, t=48, b=12), height=420,
                                          coloraxis_showscale=False)
            st.plotly_chart(attribution_fig, use_container_width=True)
            st.caption("Attributions describe how this trained model responds to the supplied inputs relative to a zero-standardised baseline. They are not causal effects.")
        except Exception as exc:
            st.warning(f"The explanation could not be generated: {exc}")

        st.subheader("Optional generative shift-handover note")
        st.write("A language model can turn the ANN forecast and available sensor context into a short operational handover note. It does not produce or modify the numerical forecast.")
        if os.getenv("OPENAI_API_KEY") and os.getenv("GENAI_MODEL"):
            if st.button("Generate handover note"):
                try:
                    from airquality.genai import generate_handover_summary
                    last = st.session_state["last_history"].iloc[-1].to_dict()
                    note, details = generate_handover_summary(prediction, TARGET_UNIT,
                        metadata["validation_ann"]["MAE"], last)
                    st.markdown(note)
                    st.caption(f"Model: {details['model']} · latency: {details['latency_seconds']} s · tokens: {details['total_tokens']}")
                except Exception as exc:
                    st.error(f"Generative summary unavailable: {exc}")
        else:
            st.info("Optional feature disabled. Set OPENAI_API_KEY and GENAI_MODEL (a currently supported model ID) in your environment to enable it. No API call is made otherwise.")

with diagnostics_tab:
    st.subheader("Held-out test performance")
    metrics_path = ROOT / "outputs" / "tables" / "final_metrics.json"
    predictions_path = ROOT / "outputs" / "tables" / "test_predictions.csv"
    comparison_path = ROOT / "outputs" / "tables" / "validation_model_comparison.csv"
    if metrics_path.exists() and predictions_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        predictions = pd.read_csv(predictions_path, parse_dates=["timestamp"])
        ann_metrics = metrics["ANN"]
        baseline_metrics = metrics["HistGradientBoosting"]
        improvement = (baseline_metrics["MAE"] - ann_metrics["MAE"]) / baseline_metrics["MAE"] * 100
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("ANN test MAE", f"{ann_metrics['MAE']:.2f} {TARGET_UNIT}")
        k2.metric("ANN test RMSE", f"{ann_metrics['RMSE']:.2f} {TARGET_UNIT}")
        k3.metric("ANN test R²", f"{ann_metrics['R2']:.3f}")
        k4.metric("MAE vs baseline", f"{improvement:.1f}% lower")
        st.caption(f"One-time held-out evaluation · {ann_metrics['n']:,} test examples. Test results are for reporting only and must not be used for further model selection.")

        left, right = st.columns(2)
        with left:
            comparison = pd.DataFrame([
                {"Model": "Initial ANN", **metrics["Initial_ANN"]},
                {"Model": "Final ANN", **metrics["ANN"]},
                {"Model": "HistGradientBoosting", **metrics["HistGradientBoosting"]},
                {"Model": "Training-mean reference", **metrics["Training_Mean"]},
            ])
            metric_choice = st.selectbox("Compare models by", ["MAE", "RMSE", "R2"], key="diagnostic_metric")
            comparison_fig = px.bar(comparison, x="Model", y=metric_choice, color="Model",
                                    title=f"Model comparison · {metric_choice}", text_auto=".3f",
                                    template="plotly_white",
                                    color_discrete_sequence=["#087E8B", "#FF5A5F", "#334E68", "#A7B8B8"])
            comparison_fig.update_layout(showlegend=False, margin=dict(l=12, r=12, t=48, b=12), height=380)
            st.plotly_chart(comparison_fig, use_container_width=True)
        with right:
            scatter_fig = px.scatter(predictions, x="actual", y="ann_prediction", color="missing_input_fraction",
                                     hover_data=["timestamp", "ann_residual"],
                                     title="ANN predictions vs actual · held-out test",
                                     labels={"actual": f"Actual ({TARGET_UNIT})", "ann_prediction": f"Predicted ({TARGET_UNIT})",
                                             "missing_input_fraction": "Missing input fraction"},
                                     template="plotly_white", color_continuous_scale=["#087E8B", "#FF5A5F"])
            bounds = [float(min(predictions["actual"].min(), predictions["ann_prediction"].min())),
                      float(max(predictions["actual"].max(), predictions["ann_prediction"].max()))]
            scatter_fig.add_shape(type="line", x0=bounds[0], y0=bounds[0], x1=bounds[1], y1=bounds[1],
                                  line=dict(color="#334E68", dash="dash"))
            scatter_fig.update_layout(margin=dict(l=12, r=12, t=48, b=12), height=380)
            st.plotly_chart(scatter_fig, use_container_width=True)

        timeline = px.line(predictions, x="timestamp", y=["actual", "ann_prediction", "baseline_prediction"],
                           title="Forecast timeline · actual and model predictions",
                           labels={"timestamp": "Time", "value": f"Benzene ({TARGET_UNIT})", "variable": "Series"},
                           template="plotly_white", color_discrete_sequence=["#334E68", "#FF5A5F", "#087E8B"])
        timeline.update_layout(margin=dict(l=12, r=12, t=48, b=12), height=420, legend_title_text="")
        st.plotly_chart(timeline, use_container_width=True)

        predictions = predictions.assign(absolute_ann_error=predictions["ann_residual"].abs())
        error_hist = px.histogram(predictions, x="ann_residual", nbins=36, marginal="box",
                                  title="ANN test residual distribution",
                                  labels={"ann_residual": f"Actual − predicted ({TARGET_UNIT})"},
                                  template="plotly_white", color_discrete_sequence=["#087E8B"])
        error_hist.add_vline(x=0, line_dash="dash", line_color="#FF5A5F")
        error_hist.update_layout(margin=dict(l=12, r=12, t=48, b=12), height=360)
        st.plotly_chart(error_hist, use_container_width=True)

        if comparison_path.exists():
            validation = pd.read_csv(comparison_path)
            validation_fig = px.bar(validation, x="model", y="MAE", color="model", text_auto=".2f",
                                    title="Validation MAE used for model selection",
                                    labels={"model": "Model", "MAE": f"MAE ({TARGET_UNIT})"},
                                    template="plotly_white",
                                    color_discrete_sequence=["#087E8B", "#FF5A5F", "#334E68", "#A7B8B8"])
            validation_fig.update_layout(showlegend=False, margin=dict(l=12, r=12, t=48, b=12), height=360)
            st.plotly_chart(validation_fig, use_container_width=True)
    else:
        st.info("Run the project pipeline to create held-out test metrics and prediction artifacts.")

with profile_tab:
    st.subheader("Training data and preprocessing profile")
    audit_path = ROOT / "outputs" / "tables" / "data_audit.json"
    missingness_path = ROOT / "outputs" / "tables" / "missingness.csv"
    monthly_path = ROOT / "outputs" / "tables" / "monthly_target_summary.csv"
    correlation_path = ROOT / "outputs" / "tables" / "correlation_matrix.csv"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Hourly rows", f"{audit['rows_after_hourly_reindex']:,}")
        a2.metric("Observed target rows", f"{audit['target_observed']:,}")
        a3.metric("Target missing", f"{audit['target_missing_pct']:.2f}%")
        a4.metric("Time coverage", f"{audit['date_start'][:10]} to {audit['date_end'][:10]}")
        st.caption("Training profile from the historical UCI station dataset. Missing sensor values are imputed using training-only medians; the fitted transformer is saved with the model artifacts.")

        profile_left, profile_right = st.columns(2)
        if missingness_path.exists():
            missingness = pd.read_csv(missingness_path).sort_values("missing_pct", ascending=True)
            missing_fig = px.bar(missingness, x="missing_pct", y="feature", orientation="h",
                                 title="Missing values by source feature",
                                 labels={"missing_pct": "Missing rows (%)", "feature": ""},
                                 template="plotly_white", color="missing_pct",
                                 color_continuous_scale=["#087E8B", "#FF5A5F"])
            missing_fig.update_layout(margin=dict(l=12, r=12, t=48, b=12), height=460,
                                      coloraxis_showscale=False)
            with profile_left:
                st.plotly_chart(missing_fig, use_container_width=True)
        if monthly_path.exists():
            monthly = pd.read_csv(monthly_path, parse_dates=["timestamp"])
            monthly_fig = px.line(monthly, x="timestamp", y=["mean", "median"], markers=True,
                                  title="Monthly target concentration",
                                  labels={"timestamp": "Month", "value": f"Benzene ({TARGET_UNIT})", "variable": "Statistic"},
                                  template="plotly_white", color_discrete_sequence=["#087E8B", "#FF5A5F"])
            monthly_fig.update_layout(margin=dict(l=12, r=12, t=48, b=12), height=460, legend_title_text="")
            with profile_right:
                st.plotly_chart(monthly_fig, use_container_width=True)
        if correlation_path.exists():
            correlation = pd.read_csv(correlation_path, index_col=0)
            corr_fig = px.imshow(correlation, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                 aspect="auto", title="Feature correlation overview",
                                 labels={"color": "Correlation"}, template="plotly_white")
            corr_fig.update_layout(margin=dict(l=12, r=12, t=48, b=12), height=640)
            st.plotly_chart(corr_fig, use_container_width=True)
            st.caption("Correlations are descriptive associations, not causal effects.")
        with st.expander("Preprocessing and leakage audit"):
            st.json(audit["leakage_audit"])
    else:
        st.info("Run the project pipeline to create the training-data profile and preprocessing audit.")
