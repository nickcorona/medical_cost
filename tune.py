from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from statsmodels.nonparametric.smoothers_lowess import lowess
import json

from helpers import encode_dates, loguniform, similarity_encode

df = pd.read_csv(
    r"data\insurance.csv",
    parse_dates=[],
    index_col=[],
    delimiter=",",
    low_memory=False,
)

print(
    pd.concat([df.dtypes, df.nunique() / len(df)], axis=1)
    .rename({0: "dtype", 1: "proportion unique"}, axis=1)
    .sort_values(["dtype", "proportion unique"])
)

TARGET = "charges"
TARGET_LEAKAGE = []
y = df[TARGET]
X = df.drop(
    [TARGET, *TARGET_LEAKAGE],
    axis=1,
)

obj_cols = X.select_dtypes("object").columns
nunique = X[obj_cols].nunique()
prop_unique = (X[obj_cols].nunique() / len(df)).sort_values(
    ascending=False
)  # in order of most unique to least
unique = pd.concat([prop_unique, nunique], axis=1)
unique.columns = [
    "proportion",
    "nunique",
]
print(unique)

ENCODE = False
if ENCODE:
    X = similarity_encode(
        X,
        encode_columns=[],
        n_prototypes=5,
        train=True,
        drop_original=False,
    )

CATEGORIZE = True
if CATEGORIZE:
    X[obj_cols] = X[obj_cols].astype("category")

sns.displot(y)
plt.title("Distribution")
plt.show()

SEED = 0
SAMPLE_SIZE = 10000

Xt, Xv, yt, yv = train_test_split(
    X, y, random_state=SEED
)  # split into train and validation set
dt = lgb.Dataset(Xt, yt, free_raw_data=False)
np.random.seed(SEED)
sample_idx = np.random.choice(Xt.index, size=SAMPLE_SIZE)
Xs, ys = Xt.loc[sample_idx], yt.loc[sample_idx]
ds = lgb.Dataset(Xs, ys)
dv = lgb.Dataset(Xv, yv, free_raw_data=False)

OBJECTIVE = "regression"
METRIC = "rmse"
MAXIMIZE = False
EARLY_STOPPING_ROUNDS = 10
MAX_ROUNDS = 10000
REPORT_ROUNDS = 5

params = {
    "objective": OBJECTIVE,
    "metric": METRIC,
    "verbose": -1,
    "n_jobs": 6,
    # "num_classes": 1,
    # "tweedie_variance_power": 1.3,
}

model = lgb.train(
    params,
    dt,
    valid_sets=[dt, dv],
    valid_names=["training", "valid"],
    num_boost_round=MAX_ROUNDS,
    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    verbose_eval=REPORT_ROUNDS,
)

lgb.plot_importance(model, grid=False, max_num_features=20, importance_type="gain")
plt.show()

TUNE_ETA = False
if TUNE_ETA:
    best_etas = {"learning_rate": [], "score": []}

    for _ in range(30):
        eta = loguniform(-5, 0)
        best_etas["learning_rate"].append(eta)
        params["learning_rate"] = eta
        model = lgb.train(
            params,
            dt,
            valid_sets=[dt, dv],
            valid_names=["training", "valid"],
            num_boost_round=MAX_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        best_etas["score"].append(model.best_score["valid"][METRIC])

    best_eta_df = pd.DataFrame.from_dict(best_etas)
    lowess_data = lowess(
        best_eta_df["score"],
        best_eta_df["learning_rate"],
    )

    rounded_data = lowess_data.copy()
    rounded_data[:, 1] = rounded_data[:, 1].round(4)
    rounded_data = rounded_data[::-1]  # reverse to find first best
    # maximize or minimize metric
    if MAXIMIZE:
        best = np.argmax
    else:
        best = np.argmin
    best_eta = rounded_data[best(rounded_data[:, 1]), 0]

    # plot relationship between learning rate and performance, with an eta selected just before diminishing returns
    # use log scale as it's easier to observe the whole graph
    sns.lineplot(x=lowess_data[:, 0], y=lowess_data[:, 1])
    plt.xscale("log")
    plt.axvline(best_eta, color="orange")
    plt.title("Smoothed relationship between learning rate and metric.")
    plt.xlabel("learning rate")
    plt.ylabel(METRIC)
    plt.show()

    print(f"Good learning rate: {best_eta:4f}")
    params["learning_rate"] = best_eta
    model = lgb.train(
        params,
        dt,
        valid_sets=[dt, dv],
        valid_names=["training", "valid"],
        num_boost_round=MAX_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=REPORT_ROUNDS,
    )
else:
    params["learning_rate"] = 0.009875772374731435

DROP_CORRELATED = False
if DROP_CORRELATED:
    threshold = 0.75
    corr = Xt.corr(method="kendall")
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    upper = upper.stack()
    high_upper = upper[(abs(upper) > threshold)]
    abs_high_upper = abs(high_upper).sort_values(ascending=False)
    pairs = abs_high_upper.index.to_list()
    correlation = len(pairs) > 0
    print(f"Correlated features: {pairs if correlation else None}")

    correlated_features = set()
    if correlation:
        # drop correlated features
        best_score = model.best_score["valid"][METRIC]
        print(f"starting score: {best_score:.4f}")
        drop_dict = {pair: [] for pair in pairs}
        for pair in pairs:
            for feature in pair:
                drop_set = correlated_features.copy()
                drop_set.add(feature)
                Xt, Xv, yt, yv = train_test_split(
                    X.drop(drop_set, axis=1), y, random_state=SEED
                )
                Xs, ys = Xt.loc[sample_idx], yt.loc[sample_idx]
                dt = lgb.Dataset(
                    Xt,
                    yt,
                    silent=True,
                )
                dv = lgb.Dataset(
                    Xv,
                    yv,
                    silent=True,
                )
                drop_model = lgb.train(
                    params,
                    dt,
                    valid_sets=[dt, dv],
                    valid_names=["training", "valid"],
                    num_boost_round=MAX_ROUNDS,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose_eval=False,
                )
                drop_dict[pair].append(drop_model.best_score["valid"][METRIC])
            pair_min = np.min(drop_dict[pair])
            if pair_min < best_score:
                drop_feature = pair[
                    np.argmin(drop_dict[pair])
                ]  # add to drop_feature the one that reduces score
                best_score = pair_min
                correlated_features.add(drop_feature)
        print(
            f"dropped features: {correlated_features if len(correlated_features) > 0 else None}"
        )
        print(f"ending score: {best_score:.4f}")

        correlation_elimination = len(correlated_features) > 0
        if correlation_elimination:
            X = X.drop(correlated_features, axis=1)
            Xt, Xv, yt, yv = train_test_split(
                X, y, random_state=SEED
            )  # split into train and validation set
            Xs, ys = Xt.loc[sample_idx], yt.loc[sample_idx]
            dt = lgb.Dataset(Xt, yt, silent=True)
            ds = lgb.Dataset(Xs, ys, silent=True)
            dv = lgb.Dataset(Xv, yv, silent=True)

            model = lgb.train(
                params,
                dt,
                valid_sets=[dt, dv],
                valid_names=["training", "valid"],
                num_boost_round=MAX_ROUNDS,
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose_eval=False,
            )
else:
    correlated_features = set()

DROP_UNIMPORTANT = False
if DROP_UNIMPORTANT:
    sorted_features = [
        feature
        for _, feature in sorted(
            zip(model.feature_importance(importance_type="gain"), dt.feature_name),
            reverse=False,
        )
    ]
    best_score = model.best_score["valid"][METRIC]
    print(f"starting score: {best_score:.4f}")
    unimportant_features = []
    for feature in sorted_features:
        unimportant_features.append(feature)
        Xt, Xv, yt, yv = train_test_split(
            X.drop(unimportant_features, axis=1), y, random_state=SEED
        )
        Xs, ys = Xt.loc[sample_idx], yt.loc[sample_idx]
        dt = lgb.Dataset(Xt, yt, silent=True)
        dv = lgb.Dataset(Xv, yv, silent=True)

        drop_model = lgb.train(
            params,
            dt,
            valid_sets=[dt, dv],
            valid_names=["training", "valid"],
            num_boost_round=MAX_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        score = drop_model.best_score["valid"][METRIC]
        if score > best_score:
            del unimportant_features[-1]  # remove from drop list
            print(f"Dropping {feature} worsened score to {score:.4f}.")
            break
        else:
            best_score = score
    print(f"ending score: {best_score:.4f}")
    print(
        f"dropped features: {unimportant_features if len(unimportant_features) > 0 else None}"
    )
    feature_elimination = len(unimportant_features) > 0

    if feature_elimination:
        X = X.drop(unimportant_features, axis=1)
        Xt, Xv, yt, yv = train_test_split(
            X, y, random_state=SEED
        )  # split into train and validation set
        Xs, ys = Xt.loc[sample_idx], yt.loc[sample_idx]
        dt = lgb.Dataset(Xt, yt, silent=True)
        ds = lgb.Dataset(Xs, ys, silent=True)
        dv = lgb.Dataset(Xv, yv, silent=True)

        model = lgb.train(
            params,
            dt,
            valid_sets=[dt, dv],
            valid_names=["training", "valid"],
            num_boost_round=MAX_ROUNDS,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=REPORT_ROUNDS,
        )
else:
    unimportant_features = []

dropped_features = list(correlated_features) + unimportant_features
print(dropped_features)

lgb.plot_importance(model, grid=False, max_num_features=20, importance_type="gain")
plt.show()

TUNE_HYPER = False
if TUNE_HYPER:
    from optuna.integration.lightgbm import LightGBMTuner

    dt = lgb.Dataset(Xt, yt, silent=True)
    ds = lgb.Dataset(Xs, ys, silent=True)
    dv = lgb.Dataset(Xv, yv, silent=True)

    auto_booster = LightGBMTuner(
        params,
        dt,
        valid_sets=[dt, dv],
        valid_names=["training", "valid"],
        num_boost_round=MAX_ROUNDS,
        verbose_eval=False,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    auto_booster.run()

    score = auto_booster.best_score
    best_params = auto_booster.best_params
    model = auto_booster.get_best_booster()
    best_params["num_boost_rounds"] = model.best_iteration
    print("Best params:", best_params)
    print(f"  {METRIC} = {score}")
    print("Params:")
    print(json.dumps(best_params, indent=4))
    print(dropped_features)
else:
    dt = lgb.Dataset(Xt, yt, silent=True)
    ds = lgb.Dataset(Xs, ys, silent=True)
    dv = lgb.Dataset(Xv, yv, silent=True)
    # rmse: 3831
    best_params = {
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
    model = lgb.train(
        best_params,
        dt,
        valid_sets=[dt, dv],
        valid_names=["training", "valid"],
        num_boost_round=MAX_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=REPORT_ROUNDS,
    )
    score = model.best_score["valid"][METRIC]
    print(f"{METRIC}: {score:.4f}")


lgb.plot_importance(
    model, grid=False, max_num_features=20, importance_type="gain", figsize=(10, 5)
)
figure_path = Path("figures")
figure_path.mkdir(exist_ok=True)
plt.savefig(figure_path / "feature_importance.png")
