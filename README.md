---
title: BeanScope - Dry Bean Classification
description: Five-classifier machine learning study with reproducible evaluation and a Streamlit app
---

## a. Problem statement

Predict one of seven dry bean varieties from 16 numerical shape and size measurements.
Compare Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes,
and Support Vector Machine on exactly the same held-out examples. Demonstrate inference
and evaluation in Streamlit without training on uploaded test data.

This is a learning reference, not an unchanged submission. The assignment permits AI
for learning support only. Understand the pipelines and metrics, make your own
reasoned adaptations, and run the project on BITS Virtual Lab. Do not invent a
commit history or use local screenshots as evidence of lab execution.

## b. Dataset description

The [UCI Dry Bean dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)
contains 13,611 examples, 16 input features, and seven target classes. It exceeds the
assignment minimums of 500 instances and 12 features. It has no missing values.
All inputs are numerical, so the project can focus on the required classifiers
rather than categorical encoding. Its unequal class sizes motivate macro-averaged
metrics alongside accuracy and MCC.

* Source: UCI Machine Learning Repository, dataset 602, donated in 2020
* Citation: Dry Bean [Dataset]. (2020). UCI Machine Learning Repository.
  [DOI: 10.24432/C50S4B](https://doi.org/10.24432/C50S4B)
* Original study: Koklu, M. and Ozkan, I. A. (2020), "Multiclass classification of
  dry beans using computer vision and machine learning techniques",
  Computers and Electronics in Agriculture, 174, 105507,
  [DOI: 10.1016/j.compag.2020.105507](https://doi.org/10.1016/j.compag.2020.105507)
* License: [Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/)
* Target column: `Class`
* Labels: BARBUNYA, BOMBAY, CALI, DERMASON, HOROZ, SEKER, SIRA
* 68 exact duplicate labeled rows removed before splitting
* 13,543 unique samples: 10,834 training rows and 2,709 test rows
* Split: 80/20, stratified on the target, `random_state=42`

The original UCI workbook is included as [the source archive](data/dry-bean-source.zip).
Its SHA-256 checksum is verified before training:

```text
0a64eff5be87f48c3dbbfc0a12a56c5d5b5167ef8e61cd45d69b3e7c7130c06f
```

### Input schema

| Group                | Features                                                            |
|----------------------|---------------------------------------------------------------------|
| Size                 | Area, Perimeter, MajorAxisLength, MinorAxisLength, ConvexArea         |
| Ratios and shape     | AspectRation, Eccentricity, EquivDiameter, Extent, Solidity           |
| Compactness and form | roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4 |

`AspectRation` and `roundness` deliberately retain the workbook's original spelling
and capitalization. The target is never included among input features.

### Held-out class distribution

| Class    | Training | Test |
|----------|----------|------|
| BARBUNYA | 1057     | 265  |
| BOMBAY   | 418      | 104  |
| CALI     | 1304     | 326  |
| DERMASON | 2837     | 709  |
| HOROZ    | 1488     | 372  |
| SEKER    | 1621     | 406  |
| SIRA     | 2109     | 527  |

## c. GitHub Repository Link

Repository: [Esha1033/dry-bean-ml-assignment](https://github.com/Esha1033/dry-bean-ml-assignment)

Live Streamlit Community Cloud app: STREAMLIT_APP_URL

The GitHub repository is published. `STREAMLIT_APP_URL` remains a placeholder;
replace it after deploying the app. The PDF builder can insert the supplied
real links into this README.

## d. Models used

Each saved artifact is a complete scikit-learn `Pipeline` containing a
`StandardScaler` and its classifier. Scaling is fitted only on the training data.
It is important for Logistic Regression, KNN and SVM; applying it consistently does
not add useful information to the tree. The same cleaned dataset and split are
used for every model.

| Model                  | Fixed configuration                                         |
|------------------------|-------------------------------------------------------------|
| Logistic Regression    | C=1.0, max_iter=3000, random_state=42                         |
| Decision Tree          | max_depth=12, min_samples_leaf=3, random_state=42             |
| KNN                    | n_neighbors=7, uniform weights, Euclidean distance           |
| Naive Bayes            | GaussianNB, default variance smoothing                      |
| Support Vector Machine | RBF kernel, C=3.0, gamma=scale, probability=True, seed=42    |

No grid search or hyperparameter changes were made using the test results.
SVC probability estimation uses its internal training-only cross-validation.

### Metric comparison

The following are measured results on the included
[held-out CSV](test_data.csv), not illustrative values.
The unrounded metrics are available in [comparison.csv](results/comparison.csv).

| ML Model Name          | Accuracy | AUC      | Precision | Recall   | F1       | MCC      |
|------------------------|----------|----------|-----------|----------|----------|----------|
| Logistic Regression    | 0.919158 | 0.994717 | 0.930725  | 0.930049 | 0.930237 | 0.902314 |
| Decision Tree          | 0.904393 | 0.970300 | 0.918241  | 0.918191 | 0.918128 | 0.884388 |
| KNN                    | 0.911776 | 0.985250 | 0.925280  | 0.922400 | 0.923654 | 0.893292 |
| Naive Bayes            | 0.897010 | 0.991143 | 0.908075  | 0.908248 | 0.907220 | 0.876171 |
| Support Vector Machine | 0.923957 | 0.994430 | 0.935927  | 0.934556 | 0.935202 | 0.908019 |

Precision, recall and F1 use unweighted macro averaging over the fixed seven-class
label set. AUC uses macro one-vs-rest ROC AUC from predicted class probabilities,
in the model's `classes_` order. MCC is the multiclass Matthews correlation
coefficient. Accuracy is the fraction of correct predictions.

For uploads missing one or more classes, seven-class AUC is undefined and is
displayed as `N/A` with a warning. Precision, recall and F1 retain all seven labels,
assigning zero to undefined class terms. Such scores are not directly comparable
to the full held-out benchmark. For a degenerate single-class prediction/truth
pair, MCC follows scikit-learn's convention of zero.

### Observations about model performance

| ML Model Name          | Observation about model performance |
|------------------------|-------------------------------------|
| Logistic Regression    | Accuracy 91.92% and macro F1 0.930237 provide a strong linear baseline. Its AUC 0.994717 is the highest, although SVM makes more correct final class decisions. SIRA F1 is 0.871028, lower than its DERMASON F1 of 0.916607. |
| Decision Tree          | Accuracy 90.44% and macro F1 0.918128 trail Logistic Regression, KNN and SVM. SIRA F1 is 0.833814, the weakest SIRA result among these five models. The depth and leaf constraints limit complexity; this experiment alone does not establish whether the remaining errors are due to overfitting. |
| KNN                    | Accuracy 91.18% and macro F1 0.923654 place it between the tree and Logistic Regression. Its distance-based prediction depends on training-only scaling. SIRA F1 0.857143 remains below DERMASON F1 0.908451. |
| Naive Bayes            | Accuracy 89.70% and macro F1 0.907220 are the lowest overall. DERMASON recall is 0.860367 while SIRA recall is 0.895636. Conditional independence is a restrictive assumption for related geometric measurements; that is a plausible explanation, not a causal result established here. Its high AUC does not imply the best class decisions. |
| Support Vector Machine | Best accuracy 92.40%, macro F1 0.935202 and MCC 0.908019. The RBF model supports nonlinear decision boundaries. SIRA F1 0.872280 still leaves room for improvement, so the highest overall score does not mean uniformly strong performance on every class. |
| Overall winner         | Support Vector Machine, using highest held-out macro F1 as the declared descriptive comparison criterion. Logistic Regression is a close alternative and has the highest AUC. No confidence intervals or significance claims are made from this single split. |

All five classifiers achieve F1 1.0 on the 104 held-out BOMBAY examples. This does
not establish perfect generalization to future BOMBAY samples. SIRA remains a
more difficult class in these measurements.

## Run locally or on BITS Virtual Lab

Use Python 3.11 for compatibility with the saved model artifacts.

```powershell
cd C:\Project\dry-bean-ml-assignment
uv sync --frozen
uv run --frozen streamlit run app.py
```

Alternatively, create a Python 3.11 environment and install the deployment
dependencies with `pip install -r requirements.txt`, then use
`streamlit run app.py`. The included saved models let the application run without
retraining or downloading data.

### Train on a VM

To run [train.py](train.py) on a VM, copy these files while preserving their
relative locations:

```text
dry-bean-ml-assignment/
  train.py
  ml_core.py
  requirements.txt
  data/
    dry-bean-source.zip
```

[ml_core.py](ml_core.py) supplies the schema, validation and evaluation functions.
Keep the [source ZIP](data/dry-bean-source.zip) unchanged in the `data` subfolder:
training verifies its checksum and reads the workbook directly from the archive.
An extracted workbook or the training CSV is not a substitute.

Use a Python 3.11 environment on the VM. From the copied project directory, install
the dependencies and start training:

```text
python -m pip install -r requirements.txt
python train.py
```

If you use `uv`, also copy [pyproject.toml](pyproject.toml), [uv.lock](uv.lock)
and [.python-version](.python-version), then run:

```powershell
uv run --frozen train.py
```

Existing files under `model` and `results`, [train_data.csv](data/train_data.csv)
and [test_data.csv](test_data.csv) are not required as training inputs.
The script creates the `model` and `results` directories when needed, but the
`data` directory and source ZIP must already exist. The project directory must be
writable. Outputs are saved beside the scripts, regardless of your working directory.

Training regenerates all five saved pipelines, the split CSVs, comparison table,
per-model classification reports and confusion matrices, and
[experiment metadata](results/metadata.json). If you alter the training protocol,
update this README's measured tables and observations from your new results.

### Git ignore policy

[.gitignore](.gitignore) excludes virtual environments, Python and test caches,
local secrets, logs, build outputs and the generated submission PDF.
Keep the original source ZIP, split CSVs, saved models, evaluation results,
dependency manifests and lockfile in version control so training and deployment
remain reproducible. Do not copy your local virtual environment to the VM;
install dependencies there instead.

## Streamlit features

* Dropdown selection for all five saved classifiers
* Bundled held-out data or labeled test CSV upload
* Accuracy, AUC, precision, recall, F1 and MCC cards
* Side-by-side results for all five models on the same uploaded rows
* Confusion matrix and per-class classification report
* Class distribution and per-row predictions
* Downloads for test data, comparison metrics and predictions
* Validation of missing, extra or duplicate columns, nonnumeric inputs,
  missing/infinite values, malformed CSV and unknown target labels
* No model fitting in the app or on uploaded examples

Uploads must contain the 16 features and `Class`; reordered columns are supported.
Extra columns are rejected. Only upload genuinely held-out test data, not the
training CSV. The limit is 20,000 rows and 10 MB.
The app does not load uploaded pickle or joblib files.

## Project files

```text
app.py                    Streamlit inference and evaluation UI
ml_core.py                Schema validation and shared evaluation
train.py                  Reproducible offline training
build_submission.py       Final PDF builder requiring real links and lab screenshot
requirements.txt          Pinned runtime dependencies for deployment
pyproject.toml            Runtime and development dependencies
uv.lock                   Reproducible environment lock
.python-version           Python 3.11 environment selection
.gitignore                Local environment, cache and secret exclusions
test_data.csv             Labeled held-out examples for upload
data/
  dry-bean-source.zip     Original attributed UCI data
  train_data.csv          Training partition; do not upload for evaluation
model/
  *.joblib                Five complete fitted preprocessing/classifier pipelines
results/
  comparison.csv         Six metrics for every classifier
  metadata.json          Split counts, checksum and experiment details
  *_confusion.csv        Per-class confusion matrices
  *_report.csv           Per-class classification reports
  app-desktop.png        Local browser screenshot, not BITS evidence
  app-mobile.png         Local browser screenshot, not BITS evidence
  browser-checks.json    Real Playwright coverage summary
tests/
  test_ml_core.py         Data, metric, pipeline and no-retraining tests
  test_browser.py         Live Streamlit tests in Chromium
  test_build_submission.py
```

Only load the joblib models supplied by your own trusted repository. Keep the
scikit-learn version pinned when loading these artifacts.

## Automated and Playwright verification

```powershell
uv run --frozen ruff check .
uv run --frozen ruff format --check .
uv run --frozen pytest tests\test_ml_core.py tests\test_build_submission.py -q
uv run --frozen playwright install chromium
$env:RUN_BROWSER_TESTS = "1"
uv run --frozen pytest tests\test_browser.py -q
```

Browser tests start and stop their own local Streamlit process. They verify every
classifier's six displayed metrics against the offline results, diagnostics,
real CSV upload, comparison/prediction downloads, invalid upload recovery,
single-class AUC handling, and desktop/mobile rendering. Local browser screenshots
are evidence of UI testing only, not BITS Virtual Lab execution.

## Deploy on Streamlit Community Cloud

1. Review and adapt the learning reference; run the experiment on BITS Virtual Lab.
2. Create your own GitHub repository and upload the project, including all five
   saved files under `model`, `test_data.csv` and `requirements.txt`.
3. Keep `.venv`, caches and secrets out of the repository.
4. Open [Streamlit Community Cloud](https://streamlit.io/cloud), sign in with GitHub,
   select your repository and branch, and set the entry point to `app.py`.
5. In Advanced settings, choose Python 3.11 before deployment.
6. Deploy and verify the live app with the included test CSV and all five models.
7. Record the real public GitHub and `.streamlit.app` URLs in this README.

The GitHub repository is published. Streamlit deployment still requires account
authorization and a verified live app URL.

## Final submission PDF

The assignment requires one PDF in this order:

1. Clickable GitHub repository link
2. Clickable live Streamlit Community Cloud app link
3. One genuine screenshot of execution on BITS Virtual Lab
4. The complete GitHub README content, including metrics and observations

After you have the real links and the lab screenshot, run:

```powershell
uv run --frozen python build_submission.py `
  --github-url "https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY" `
  --app-url "https://YOUR-APP.streamlit.app" `
  --lab-screenshot "C:\path\to\your-real-bits-lab-screenshot.png"
```

The builder creates `submission.pdf` and replaces the two link tokens in this
README. Publish the updated README afterward. It checks URL shapes and image
readability, but it cannot authenticate screenshot provenance or guarantee public
link accessibility: verify those yourself. Do not pass the local UI screenshots
as lab evidence. The PDF builder requires the development dependencies installed
by `uv sync`, including ReportLab.

Before submitting, open both public links in a signed-out browser, test the live
app, verify all PDF pages and evidence, and submit the final PDF rather than a
draft. The supplied assignment states a deadline of 8 November 2026 at 23:59.
