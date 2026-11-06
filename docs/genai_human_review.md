# Generative-AI handover note: human review rubric

Use this review only after the lecturer approves the adjacent text-generation task and the API evaluation is run. The LLM must not be given the future actual benzene target.

For every generated test-case note, score each dimension from 1 (poor) to 5 (excellent):

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Factuality | Contains material unsupported or contradictory claims | Mostly grounded, with one notable unsupported detail | All stated facts match the supplied ANN forecast/context |
| Completeness | Omits the forecast or key caution | Includes forecast and some limitations | Includes forecast, origin/context, key data-quality caveat and reference-check caveat |
| Clarity | Ambiguous or difficult to act on | Understandable but wordy or incomplete | Concise, specific and easy for a shift operator to interpret |
| Unsupported assertion flag | Mark Yes if it invents a cause, threshold, safety claim, event or action not supported by the inputs | | |

Record results in `outputs/tables/genai_human_review_template.csv`. Include reviewer initials/notes and examples of any hallucination. Report the number of cases reviewed and observed scores; do not present an unreviewed generated note as validated operational advice.

This is a qualitative assessment of an adjacent writing task, not a substitute for regression metrics and not a claim that the language model predicts benzene.
