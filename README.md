# GBMSurv: Second-Order Gradient Boosting for Survival Analysis

GBMSurv is a collection of gradient boosting-based models for survival analysis. It leverages gradient boosting techniques to predict survival times and perform risk analysis for censored survival data. The model is particularly useful for predicting time-to-event data, which is common in medical research, reliability analysis, and various engineering applications. This repository contains implementations of all model variants.

## Installation

To ensure compatibility and avoid conflicts, it is recommended to set up an isolated Python environment. You can use [conda](https://docs.anaconda.com/miniconda/) for this purpose.

To install `GBMSurv` in development mode after cloning the repository, follow these steps:

```bash
git clone https://github.com/NTAILab/GBMSurv.git
cd GBMSurv
pip install -e .
```

## Package Contents

The repository is organized into several main directories:

* **`NonParamGBMSurv/`**, **`LogNormGBMSurv/`**, **`WeibGBMSurv/`**
  These folders contain implementations of three model variants:

  * **NonParamGBMSurv** – non-parametric model that estimates event probabilities within discrete time intervals.
  * **LogNormGBMSurv** – parametric model based on the Log-Normal distribution.
  * **WeibGBMSurv** – parametric model based on the Weibull distribution.

  Each folder includes the following submodules:

  * **`model.py`** – defines the core gradient boosting survival model.
  * **`loss.py`** – contains the loss function and its gradients/Hessians adapted for survival analysis.
  * **`utils.py`** – utility functions for data preprocessing, metric calculations, and experimental setup.

* **`examples/`**
  Contains example notebooks that demonstrate how to train and evaluate the models on survival data.


## Usage

Example usage is provided in the `examples` directory, including a demonstration of the model's application to survival datasets.

To use the model for survival analysis, follow these steps:

1. Preprocess the dataset, ensuring it contains censored survival times (e.g., time-to-event data) in the format `(delta, time)` where:
   - `delta`: Censoring indicator (1 if the event occurred, 0 if the data is censored).
   - `time`: The observed survival time.
   The target variable `y` should be in the form of a structured NumPy array with the fields `delta` and `time`.

2. Define the required model using `NonParamGBMSurv`, `LogNormGBMSurv` or `WeibGBMSurv`.
3. Train the model and evaluate performance metrics, such as the C-index, for model evaluation.

Here’s an example of using `GBMSurv` models for survival analysis:

```python
from WeibGBMSurv.model import WeibGBMSurvivalModel
from NonParamGBMSurv.model import GBMSurvivalModel
from LogNormGBMSurv.model import LogNormGBMSurvivalModel

from sksurv.datasets import load_veterans_lung_cancer
from sklearn.model_selection import train_test_split

X, y = load_veterans_lung_cancer()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

weib = WeibGBMSurvivalModel(n_estimators=100, max_depth=5, learning_rate=0.1)
lognorm = LogNormGBMSurvivalModel(n_estimators=100, max_depth=5, learning_rate=0.1)
gbmsurv = GBMSurvivalModel(n_estimators=100, max_depth=5, learning_rate=0.1)

weib.fit(X_train, y_train)
lognorm.fit(X_train, y_train)
gbmsurv.fit(X_train, y_train)

preds_weib = weib.predict(X_test)
preds_lognorm = lognorm.predict(X_test)
preds_gbmsurv = gbmsurv.predict(X_test)

c_index_weib = weib.score(X_test, y_test)
c_index_lognorm = lognorm.score(X_test, y_test)
c_index_gbmsurv = gbmsurv.score(X_test, y_test)

print(f'C-index (Weibull): {c_index_weib}')
print(f'C-index (Log-Normal): {c_index_lognorm}')
print(f'C-index (Non-Parametric): {c_index_gbmsurv}')
```

This will train the `NonParamGBMSurv`, `LogNormGBMSurv` and `WeibGBMSurv` models on Veterans dataset and provide predictions on test data.

## Citation

If you use this project in your research, please cite it as follows:

...will be later.