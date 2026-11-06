# Air Quality Intelligence – GenAI Shift-Handover Evaluation

## Lecturer approval
The user has confirmed lecturer approval for this adjacent GenAI task.

## Install / configure
1. Keep `genai.py` at `src/airquality/genai.py`.
2. Keep `evaluate_genai.py` at `scripts/evaluate_genai.py`.
3. Keep `shift_handover_v1.txt` at `prompts/shift_handover_v1.txt`.
4. Ensure project dependencies include `openai`, `python-dotenv`, `pandas`, and `numpy`.
5. Copy `.env.example` to `.env` in the project root. Put the replacement API key in `.env` locally and set `GENAI_MODEL` to a model enabled for the API project. Never share `.env` or commit it.
6. Run from the project root: `python scripts/evaluate_genai.py --n 5`.

## Outputs created by a successful real API run
- `outputs/genai/genai_handover_evaluation.json`
- `outputs/tables/genai_evaluation_log.csv`
- `outputs/tables/genai_human_review.csv`
- `outputs/tables/genai_cost_summary.csv`

The script samples evenly spaced rows from the held-out prediction file. It sends only the ANN forecast, validation MAE, timestamp and allow-listed recent sensor context. It does not send the future actual target, residuals or baseline predictions.

## Important evaluation distinction
The LLM produces text, not a numerical benzene forecast. Do not compare it with MAE/RMSE/R². Evaluate the notes for factuality, completeness, clarity, unsupported assertions, latency and token/cost use. Human review is required; the script intentionally leaves human scores blank.

## Security
A previously shared API key must be revoked. Use a newly created key only in your local `.env` file. Do not send it in chat, put it in source code, or upload it with project files.
