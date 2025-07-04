# train_and_export_model.py
# ────────────────────────────────────────────────────────────
import pandas as pd
import joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# 1. Load dataset yang MASIH punya kolom workclass
df = pd.read_csv("dataset/train_mod3.csv")     # pastikan file ini berisi kolom workclass

# 2. Drop kolom yang tidak dipakai
drop_cols = ["fnlwgt", "education", "native-country", "occupation", "age_group"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 3. Target
y = (df["income"] == ">50K").astype(int)
X = df.drop(columns=["income"])

# 4. Tentukan kolom numerik & kategorikal
num_cols = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
cat_cols = [
    "workclass", "marital-status", "occupation_grouped",
    "relationship", "race", "sex", "native_region"
]

# 5. Preprocessing
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("ohe", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# 6. Pipeline
pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", XGBClassifier(
        n_estimators=250,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        random_state=42,
        eval_metric="logloss"
    ))
])

pipe.fit(X, y)

# 7. Save pipeline
Path("models").mkdir(exist_ok=True)
joblib.dump(pipe, "models/income_xgb.pkl")
print("✅  Pipeline (with workclass) saved to models/income_xgb.pkl")
