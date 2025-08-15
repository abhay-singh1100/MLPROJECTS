# MLPROJECT — End-to-End Machine Learning Pipeline

This repository contains an end-to-end machine learning project that demonstrates how to move from notebook exploration to a reproducible, production-oriented training pipeline.

Key features
- Modular components for data ingestion, transformation, and model training located in `src/components`.
- Notebooks for EDA and experimentation under `notebook/` and `notebook12/`.
- Artifact management (preprocessors, datasets, trained model) under `artifacts/` and `artifacts0/`.
- Hyperparameter tuning support added to the model trainer using `GridSearchCV` (see `src/components/model_trainer.py` and `src/utils.py`).
- Logging and a simple custom exception wrapper in `src/logger.py` and `src/exception.py`.

What changed recently
- Added hyperparameter tuning: `evaluate_models` now accepts parameter grids and returns fitted/tuned estimators. The trainer runs GridSearchCV for models with provided grids and selects the best model by test R2 score.
- CatBoost configuration: to run cross-validation / GridSearch without trying to create local `catboost_info` folders, CatBoost is instantiated with `allow_writing_files=False` in `src/components/model_trainer.py`.

Repository structure (important files)

- `src/components/data_ingestion.py` — code to read raw data and split/save train/test.
- `src/components/data_transformation.py` — preprocessing and pipeline construction; produces a preprocessor object saved to `artifacts/preprocessor.pkl`.
- `src/components/model_trainer.py` — trains multiple models, performs hyperparameter tuning, selects and saves the best model to `artifacts0/model.pkl`.
- `src/utils.py` — helper utilities; now contains `evaluate_models` which supports GridSearchCV and returns both scores and fitted estimators.
- `notebook/` and `notebook12/` — exploratory notebooks and model-training notebooks used during development.
- `requirements.txt` — runtime dependencies for the project environment.

Quick start

1. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Prepare data and run ingestion/transformation (example):

```powershell
python src\components\data_ingestion.py
python src\components\data_transformation.py
```

3. Run training (this will perform hyperparameter tuning where configured):

```powershell
python -c "from src.components.model_trainer import ModelTrainer; import numpy as np;\
train = np.load('artifacts/train.npy', allow_pickle=True); test = np.load('artifacts/test.npy', allow_pickle=True);\
mt = ModelTrainer(); print(mt.initiate_model_trainer(train, test))"
```

Notes and troubleshooting
- If GridSearch with CatBoost raises errors about writing to `catboost_info`, the trainer already sets `allow_writing_files=False` to avoid creation of local folders during CV. If you still see errors, ensure the running process has write permissions to the working directory or adjust CatBoost params.
- The tuning grids are intentionally small for CI-speed; expand them in `src/components/model_trainer.py` for more thorough searches.

Contributing
- Feel free to open issues or PRs. Suggestions: add RandomizedSearch, integrate with MLflow for tracking, or add unit tests for the training pipeline.

License
- MIT