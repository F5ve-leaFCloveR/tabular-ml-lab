# Model Card: Tabular ML Lab (Adult Income)

## Model Details
- **Architecture:** Multi-layer perceptron (MLP) with ReLU and dropout
- **Framework:** PyTorch
- **Task:** Binary classification (`>50K` income threshold)

## Intended Use
- Demonstrate an end-to-end tabular ML workflow
- Educational and portfolio purposes

## Training Data
- **Dataset:** UCI Adult
- **Size:** ~32k training rows, ~16k test rows
- **Features:** 14 input features (6 numeric, 8 categorical)

## Evaluation Data
- Official UCI Adult test split (`adult.test`)

## Metrics
Metrics are produced by `python scripts/evaluate.py` and stored in `reports/`.

## Ethical Considerations
- The dataset includes sensitive attributes (e.g., race, sex). Predictions can reinforce bias.
- This model is **not** suitable for real-world decisions about employment, credit, or housing.

## Limitations
- Trained on a historical dataset that may not represent current demographics.
- Performance can degrade for underrepresented groups.
- Not calibrated; outputs are raw probabilities.

## Recommendations
- Use only for experimentation and learning.
- If deployed, conduct bias audits and calibration.
