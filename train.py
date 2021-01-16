import lightgbm as lgb
from helpers import similarity_encode
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

df = pd.read_csv(
    r"data\insurance.csv",
    parse_dates=[],
    index_col=[],
)
TARGET = "charges"
DROPPED_FEATURES = []
y = df[TARGET]
X = df.drop([TARGET, *DROPPED_FEATURES], axis=1)

CATEGORIZE = True
if CATEGORIZE:
    obj_cols = X.select_dtypes("object").columns
    X[obj_cols] = X[obj_cols].astype("category")

params = {
    "objective": "regression",
    "metric": "rmse",
    "verbose": -1,
    "n_jobs": 6,
    "learning_rate": 0.009875772374731435,
    "feature_pre_filter": False,
    "lambda_l1": 1.8299011908715305e-06,
    "lambda_l2": 0.007585780742403452,
    "num_leaves": 6,
    "feature_fraction": 1.0,
    "bagging_fraction": 0.9836888816811709,
    "bagging_freq": 3,
    "min_child_samples": 20,
    "num_boost_rounds": 471,
}
d = lgb.Dataset(X, y, silent=True)
model = lgb.train(params, d)

Path("models").mkdir(exist_ok=True)
model.save_model(
    "models/model.pkl",
    importance_type="gain",
)
