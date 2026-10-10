# Contributing to LEDGERLENS

LEDGERLENS is a baseline anomaly-detection prototype for an online coding event. Open GitHub Issues define the tasks and carry an `easy`, `medium`, or `hard` difficulty label.

## Workflow

1. Read the open Issues and choose one that matches your experience.
2. Comment on the issue before starting to reduce duplicate work.
3. Fork or clone the repository and create a branch.
4. Reproduce the current Isolation Forest baseline.
5. Make a focused change related to the selected issue.
6. Run the tests and verify the app locally.
7. Open a Pull Request linking the issue (for example, `Closes #2`).

## Evaluating anomaly detection

Report the method, feature changes, preprocessing, parameter choices, anomaly rate, and examples of projects whose flags changed. Explain why the result is useful and document assumptions and limitations.

Do not report accuracy or other supervised metrics unless reliable ground-truth labels exist. Keep evaluation separate from model development and avoid data leakage. An unusual project is a candidate for human review, not proof of fraud or corruption.

## General guidelines

- Explain what changed, why, and how it was tested.
- Keep code, plots, and documentation understandable to other students.
- Do not commit secrets, virtual environments, or generated cache files.
- Remember that the starter dataset is synthetic and does not represent official government findings.

Difficulty labels describe expected scope, not guaranteed completion time.