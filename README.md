## PS5 - Hidden Markov Models (HMM)

Hidden Markov Model tagging/training for sequence labeling (e.g., predicting POS-like tags from words). Includes a debugging runner and an HMM main driver.

### Run
- Debug/demo: `src/HMMDebug.java` (class `HMMDebug`)
- Main accuracy run: `src/HMM.java` (its `main`)

### Important: update file paths
These mains reference training/test files using absolute paths. Update them to point to your local `CS10/Inputs/` copies:
- `brown-train-sentences.txt`, `brown-train-tags.txt`
- `brown-test-sentences.txt`, `brown-test-tags.txt`

