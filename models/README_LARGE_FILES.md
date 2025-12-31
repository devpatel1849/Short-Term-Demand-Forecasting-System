# Large Model Files

Due to GitHub's file size limitations, the large model files are stored separately:

## Missing Model Files:
- `manufacturing_random_forest_model.pkl` (241.99 MB)
- `advanced_ml_ensemble_random_forest.pkl` (53.94 MB)

## How to obtain the models:

### Option 1: Download from Google Drive
The complete model files are available in the project's Google Drive folder:
https://drive.google.com/drive/folders/1vjQb8DJ7Ze4BCeWl9arMJAG67twUrnVt?usp=drive_link

### Option 2: Retrain the models
Run the training script to generate the models locally:
```bash
python integrate_real_data.py
```

### Option 3: Use the dashboard without models
The dashboard includes a model training feature that will automatically create the required models on first run.

## File Placement:
After downloading, place the model files in this `models/` directory:
```
models/
├── advanced_ml_ensemble_metadata.json
├── advanced_ml_ensemble_random_forest.pkl  ← Place here
├── advanced_ml_ensemble_xgboost.pkl
├── feature_columns.txt
├── label_encoders.pkl
└── manufacturing_random_forest_model.pkl   ← Place here
```

## Note:
The remaining smaller model files (XGBoost, label encoders, etc.) are included in this repository.
