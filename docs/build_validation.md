# Build and verification status of the supplied ZIP

This file distinguishes software verification from model evaluation on the official UCI data.

## Checks completed while preparing this package

- Python source compilation: passed (`python -m compileall -q src app scripts tests`).
- Unit/integration tests: passed (4 tests) using a temporary synthetic UCI-format fixture.
- Parser test covers semicolon separation, decimal comma, timestamp parsing and `-200` missing markers.
- Data-bundle test checks temporal split boundaries, target-row exclusion, and training-input-only preprocessing fit.
- ANN test checks tensor dimensions, training execution and finite Integrated Gradients output.
- Inference test checks request-time prediction, required columns, consecutive timestamps and out-of-training-range detection.
- An end-to-end pipeline smoke test was completed separately using a generated synthetic fixture. No synthetic data, predictions or metric values are included as UCI results in this ZIP.

## Not yet completed on official observations

The official UCI archive could not be transferred into the build environment. Consequently, the package contains no official `AirQualityUCI.csv`, real-data-trained weights, final test metrics, or measured real-data experiment log. The report is explicitly a pre-run scaffold. Execute the official downloader and complete pipeline in an environment with access to UCI before submission. Do not report the synthetic tests as evidence of predictive accuracy.

## User environment checks

The Streamlit application source is included and `streamlit` is declared in `requirements.txt`; browser-based app startup was not verified in the build environment because the Streamlit executable was not installed there. Install the requirements, run the data pipeline, then launch the app as documented in `README.md`.
