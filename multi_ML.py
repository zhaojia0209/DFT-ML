# -*- coding: utf-8 -*-
"""
Function-based ML pipeline:
- RandomizedSearchCV (n_iter configurable) + KFold for hyperparameter selection
- Manual KFold CV for obtaining CV predictions/residuals
- Train on full dataset with best estimator
- Save predictions, feature importances, plots, and high-error samples
- HuberRegressor tuned to avoid long stalls

Date: 2025-10-11
"""
# -*- coding: utf-8 -*-
"""
Function-based ML pipeline optimized for small dataset (~40 samples):
- RandomizedSearchCV + KFold for hyperparameter selection
- Manual KFold CV for out-of-fold predictions/residuals
- Train on full dataset with best estimator
- Save predictions, feature importances, plots, and high-error samples
Author: optimized for small sample
Date: 2025-10-11
"""

import os
import json
import time
import warnings
from pprint import pformat

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from scipy.stats import randint, uniform, loguniform

# Models
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor, ElasticNetCV, LassoLarsIC, BayesianRidge, ARDRegression
from sklearn.kernel_ridge import KernelRidge


# ---------------------------------------------------------------------
# Settings / configuration
# ---------------------------------------------------------------------
warnings.filterwarnings("ignore", category=ConvergenceWarning)
sns.set_style("whitegrid")

class CFG:
    SEED = 30
    KFOLD = 5
    N_ITER = 20              # RandomizedSearchCV iterations
    SAVE_DIR = "results"
    PLOT_DIR = "plots"
    PARAM_DIR = "best_params"
    CV_SUMMARY_DIR = "cv_results"
    HIGH_ERROR_THRESHOLD = 200
    VERBOSE_SEARCH = 2
    N_JOBS = -1

def init_dirs():
    os.makedirs(CFG.SAVE_DIR, exist_ok=True)
    os.makedirs(CFG.PLOT_DIR, exist_ok=True)
    os.makedirs(CFG.PARAM_DIR, exist_ok=True)
    os.makedirs(CFG.CV_SUMMARY_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
def to_serializable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return o

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, default=to_serializable, indent=4)

# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
def load_data(csv_path='binary_alloy_features_extended.csv'):
    df = pd.read_csv(csv_path)
    feature_cols = [
        'chi_A','chi_B','delta_chi','r_A','r_B','delta_r','Z_A','Z_B',
        'valence_A','valence_B','delta_valence','V_atom_A','V_atom_B','delta_V',
        'EA_A','EA_B','delta_EA','IE_A','IE_B','delta_IE',
        'chi_prod','chi_avg','delta_chi_sq','delta_r_sq','chi_r_interaction','delta_r_host_sum'
    ]
    target_cols = ['E_far','E_nn']
    X = np.nan_to_num(df[feature_cols].values)
    y = np.nan_to_num(df[target_cols].values)
    return df, X, y, feature_cols, target_cols

# ---------------------------------------------------------------------
# Models optimized for small dataset
# ---------------------------------------------------------------------
def get_models():
    models = {
        # Linear / robust
        'LinearRegression': LinearRegression(),
        'Ridge': Ridge(alpha=1.0, max_iter=5000),
        'Lasso': Lasso(alpha=0.01, max_iter=5000),
        'ElasticNet': ElasticNetCV(l1_ratio=[0.1,0.5,0.9], alphas=[1e-3,1e-2,1e-1], cv=3, random_state=CFG.SEED),
        'LassoLarsIC': LassoLarsIC(criterion='aic'),
        'Huber': HuberRegressor(max_iter=200, tol=1e-3, warm_start=True),
        'KernelRidge': KernelRidge(alpha=1.0, kernel='rbf'),

        # Bayesian / small-sample
        'BayesianRidge': BayesianRidge(),
        'ARDRegression': ARDRegression(),

        # Small trees / boosting
        'RandomForest': RandomForestRegressor(n_estimators=50, max_depth=3, min_samples_leaf=2, random_state=CFG.SEED),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=50, max_depth=3, min_samples_leaf=2, random_state=CFG.SEED),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=50, max_depth=2, learning_rate=0.05, random_state=CFG.SEED),

        # Small KNN / SVR / MLP
        'KNN': KNeighborsRegressor(n_neighbors=3),
        'SVR': SVR(kernel='rbf', C=10, epsilon=0.01, gamma='scale'),
        'MLP': MLPRegressor(hidden_layer_sizes=(20,5), max_iter=2000, random_state=CFG.SEED)
    }
    return models

# ---------------------------------------------------------------------
# Parameter distributions for small dataset
# ---------------------------------------------------------------------
def get_param_distributions():
    pdist = {
        "RandomForest": {
            "estimator__n_estimators": randint(30, 100),
            "estimator__max_depth": randint(2, 5)
        },
        "ExtraTrees": {
            "estimator__n_estimators": randint(30, 100),
            "estimator__max_depth": randint(2, 5)
        },
        "GradientBoosting": {
            "estimator__n_estimators": randint(30, 100),
            "estimator__max_depth": randint(2, 4),
            "estimator__learning_rate": uniform(0.01, 0.1)
        },
        "KNN": {
            "estimator__n_neighbors": randint(2, 5)
        },
        "SVR": {
            "estimator__C": loguniform(1, 50),
            "estimator__epsilon": uniform(0.001, 0.05)
        },
        "MLP": {
            "estimator__hidden_layer_sizes": [(20,5),(30,10)],
            "estimator__alpha": loguniform(1e-5, 1e-2)
        },
        "Ridge": {"estimator__alpha": loguniform(1e-3, 10)},
        "Lasso": {"estimator__alpha": loguniform(1e-3, 1.0)},
        "Huber": {"estimator__alpha": loguniform(1e-3, 1e-1),"estimator__epsilon": uniform(1.1, 0.5)},
        "KernelRidge": {"estimator__alpha": loguniform(1e-3, 1.0)},
        "BayesianRidge": {
            "estimator__alpha_1": loguniform(1e-6, 1e-2),
            "estimator__alpha_2": loguniform(1e-6, 1e-2),
            "estimator__lambda_1": loguniform(1e-6, 1e-2),
            "estimator__lambda_2": loguniform(1e-6, 1e-2)
        },
        "ARDRegression": {
            "estimator__alpha_1": loguniform(1e-6, 1e-2),
            "estimator__alpha_2": loguniform(1e-6, 1e-2),
            "estimator__lambda_1": loguniform(1e-6, 1e-2),
            "estimator__lambda_2": loguniform(1e-6, 1e-2)
        }
    }
    return pdist



# ---------------------------------------------------------------------
# Randomized search wrapper
# ---------------------------------------------------------------------
def run_random_search(base_model, model_name, X_scaled, y, param_distributions):
    """
    base_model: base estimator instance (not MultiOutputRegressor-wrapped)
    model_name: string key
    X_scaled: scaled X (numpy array)
    y: target array
    param_distributions: dict mapping model_name -> param_dist for MultiOutputRegressor (with 'estimator__' prefix)
    Returns: best_estimator (MultiOutputRegressor), best_params (dict) or (base wrapped, {}) if skipped/failed
    """
    if model_name not in param_distributions or not param_distributions.get(model_name):
        # no search space defined
        wrapped = MultiOutputRegressor(clone(base_model), n_jobs=CFG.N_JOBS)
        return wrapped, {}

    print(f"[SEARCH] Running RandomizedSearchCV for {model_name} (n_iter={CFG.N_ITER})")
    wrapped = MultiOutputRegressor(clone(base_model), n_jobs=CFG.N_JOBS)
    search = RandomizedSearchCV(
        estimator=wrapped,
        param_distributions=param_distributions[model_name],
        n_iter=CFG.N_ITER,
        cv=KFold(n_splits=CFG.KFOLD, shuffle=True, random_state=CFG.SEED),
        scoring='neg_mean_squared_error',  # minimize MSE during search
        random_state=CFG.SEED,
        n_jobs=CFG.N_JOBS,
        verbose=CFG.VERBOSE_SEARCH,
        refit=True,
        return_train_score=False
    )
    try:
        t0 = time.time()
        search.fit(X_scaled, y)
        t1 = time.time()
        best = search.best_estimator_
        best_params = search.best_params_
        print(f"[SEARCH] Done for {model_name} in {t1 - t0:.1f}s. Best params extracted.")
        # save cv_results summary
        try:
            save_summary = {
                "best_score": float(search.best_score_),
                "best_params": best_params
            }
            save_json(save_summary, os.path.join(CFG.CV_SUMMARY_DIR, f"{model_name}_cv_summary.json"))
        except Exception:
            pass
        # save best params
        try:
            save_json(best_params, os.path.join(CFG.PARAM_DIR, f"{model_name}_best_params.json"))
        except Exception:
            pass
        return best, best_params
    except Exception as e:
        print(f"[WARN] RandomizedSearchCV failed for {model_name}: {e}. Falling back to default wrapped estimator.")
        wrapped = MultiOutputRegressor(clone(base_model), n_jobs=CFG.N_JOBS)
        return wrapped, {}

# ---------------------------------------------------------------------
# Manual KFold CV to produce out-of-fold predictions
# ---------------------------------------------------------------------
def manual_kfold_cv(estimator, X, y, n_splits=CFG.KFOLD, random_state=CFG.SEED):
    """
    estimator: MultiOutputRegressor (unfitted)
    Returns: cv_preds (n_samples, n_targets)
    """
    n_samples = X.shape[0]
    n_targets = y.shape[1]
    cv_preds = np.zeros((n_samples, n_targets))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold = 0
    for train_idx, test_idx in kf.split(X):
        fold += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        est_clone = clone(estimator)
        try:
            est_clone.fit(X_tr_s, y_train)
            preds = est_clone.predict(X_te_s)
            cv_preds[test_idx] = preds
        except Exception as e:
            print(f"[WARN] fold {fold} training/prediction failed: {e}. Filling zeros for this fold.")
            cv_preds[test_idx] = np.zeros((len(test_idx), n_targets))
    return cv_preds

# ---------------------------------------------------------------------
# Train final model on full dataset
# ---------------------------------------------------------------------
def train_on_full(estimator, X, y):
    """
    estimator: MultiOutputRegressor (either best from search or wrapped default)
    Returns: fitted_estimator, X_scaled_full, y_pred_train
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    est_clone = clone(estimator)
    try:
        est_clone.fit(X_scaled, y)
        y_pred = est_clone.predict(X_scaled)
        return est_clone, X_scaled, y_pred
    except Exception as e:
        print(f"[ERROR] Full training failed: {e}. Returning zeros for train predictions.")
        n_targets = y.shape[1]
        return est_clone, X_scaled, np.zeros((X.shape[0], n_targets))

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def compute_metrics(y_true, y_pred, target_names):
    metrics = {}
    for i, t in enumerate(target_names):
        try:
            mse = mean_squared_error(y_true[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_true[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_true[:, i], y_pred[:, i])
        except Exception as e:
            print(f"[WARN] metrics failed for target {t}: {e}")
            rmse = np.nan; r2 = np.nan; mae = np.nan
        metrics[t] = {"RMSE": float(rmse), "R2": float(r2), "MAE": float(mae)}
    return metrics

# ---------------------------------------------------------------------
# Feature importance (robust)
# ---------------------------------------------------------------------
def get_feature_importance(fitted_multi, X_scaled, y, feature_names, target_names):
    """
    Inspect each underlying estimator in MultiOutputRegressor:
    - use feature_importances_ if present
    - else use coef_
    - else fallback to permutation importance
    Returns dict: {target: {feature: importance}}
    """
    importance_dict = {}
    for i, est in enumerate(fitted_multi.estimators_):
        tname = target_names[i]
        imp = None
        try:
            if hasattr(est, "feature_importances_"):
                imp = np.array(est.feature_importances_)
            elif hasattr(est, "coef_"):
                imp = np.abs(np.ravel(est.coef_))
        except Exception:
            imp = None
        if imp is None:
            try:
                res = permutation_importance(est, X_scaled, y[:, i], n_repeats=8, random_state=CFG.SEED, n_jobs=CFG.N_JOBS)
                imp = res.importances_mean
            except Exception as e:
                print(f"[WARN] permutation importance failed for target {tname}: {e}")
                imp = np.zeros(len(feature_names))
        # ensure length matches
        if len(imp) != len(feature_names):
            arr = np.zeros(len(feature_names))
            arr[:min(len(imp), len(arr))] = np.array(imp).ravel()[:min(len(imp), len(arr))]
            imp = arr
        importance_dict[tname] = dict(zip(feature_names, imp.tolist()))
    return importance_dict

# ---------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------
def plot_residuals(cv_preds, y_true, target_names, model_name):
    for i, t in enumerate(target_names):
        residuals = y_true[:, i] - cv_preds[:, i]
        plt.figure(figsize=(6,4))
        plt.scatter(cv_preds[:, i], residuals, alpha=0.6)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel(f"Predicted ({t})")
        plt.ylabel("Residual")
        plt.title(f"{t} Residuals ({model_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(CFG.PLOT_DIR, f"{model_name}_{t}_residuals.png"), dpi=300)
        plt.close()

def plot_pred_vs_true(y_true, y_pred_train, cv_preds, target_names, model_name):
    for i, t in enumerate(target_names):
        plt.figure(figsize=(6,4))
        plt.scatter(y_true[:, i], y_pred_train[:, i], alpha=0.6, label='Train')
        plt.scatter(y_true[:, i], cv_preds[:, i], alpha=0.6, label='CV')
        mn, mx = np.nanmin(y_true[:, i]), np.nanmax(y_true[:, i])
        plt.plot([mn, mx], [mn, mx], 'k--')
        plt.xlabel(f"True ({t})")
        plt.ylabel(f"Predicted ({t})")
        plt.title(f"{t} Prediction ({model_name})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(CFG.PLOT_DIR, f"{model_name}_{t}_prediction.png"), dpi=300)
        plt.close()

def plot_feature_importance(importance_dict, feature_names, target_names, model_name, topk=10):
    for t in target_names:
        imp_map = importance_dict.get(t, {})
        if not imp_map:
            continue
        sorted_items = sorted(imp_map.items(), key=lambda x: x[1], reverse=True)
        features, values = zip(*sorted_items[:topk])
        plt.figure(figsize=(6,4))
        plt.barh(features, values)
        plt.gca().invert_yaxis()
        plt.xlabel("Importance")
        plt.title(f"{t} Top Features ({model_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(CFG.PLOT_DIR, f"{model_name}_{t}_feature_importance.png"), dpi=300)
        plt.close()

def plot_high_error_samples(df, cv_preds, model_name):
    # high error on E_far (first target)
    residuals = df['E_far'].values - cv_preds[:, 0]
    idx = np.where(np.abs(residuals) > CFG.HIGH_ERROR_THRESHOLD)[0]
    if len(idx) == 0:
        return
    high_df = df.iloc[idx].copy()
    high_df['E_far_error'] = residuals[idx]
    # save CSV
    high_df.to_csv(os.path.join(CFG.SAVE_DIR, f"{model_name}_high_error_samples.csv"), index=False)
    # scatter plot delta_r vs delta_chi if columns exist
    if 'delta_r' in df.columns and 'delta_chi' in df.columns:
        plt.figure(figsize=(6,4))
        sns.scatterplot(x='delta_r', y='delta_chi', hue='E_far_error', data=high_df, palette='coolwarm', size=np.abs(high_df['E_far_error']), sizes=(20,200))
        plt.xlabel("delta_r"); plt.ylabel("delta_chi"); plt.title(f"High-error Samples ({model_name})")
        plt.tight_layout()
        plt.savefig(os.path.join(CFG.PLOT_DIR, f"{model_name}_high_error_distribution.png"), dpi=300)
        plt.close()

# ---------------------------------------------------------------------
# Save predictions and results
# ---------------------------------------------------------------------
def save_predictions(y_true, cv_preds, y_pred_train, model_name):
    out_df = pd.DataFrame({
        'E_far_true': y_true[:, 0],
        'E_nn_true': y_true[:, 1],
        'E_far_cv_pred': cv_preds[:, 0],
        'E_nn_cv_pred': cv_preds[:, 1],
        'E_far_train_pred': y_pred_train[:, 0],
        'E_nn_train_pred': y_pred_train[:, 1]
    })
    out_df['E_far_residual'] = out_df['E_far_true'] - out_df['E_far_cv_pred']
    out_df['E_nn_residual'] = out_df['E_nn_true'] - out_df['E_nn_cv_pred']
    path = os.path.join(CFG.SAVE_DIR, f"{model_name}_predictions.csv")
    out_df.to_csv(path, index=False)

# ---------------------------------------------------------------------
# Full pipeline loop
# ---------------------------------------------------------------------
def run_full_pipeline(csv_path='binary_alloy_features_extended.csv', models_to_run=None):
    init_dirs()
    df, X, y, feature_cols, target_names = load_data(csv_path)
    print(f"[INFO] Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

    models = get_models()
    param_distributions = get_param_distributions()

    # If user passed subset of models to run
    if models_to_run is not None:
        models = {k: v for k, v in models.items() if k in models_to_run}
        print(f"[INFO] Running subset of models: {list(models.keys())}")

    results_summary = []

    # Pre-scale once for hyperparameter search convenience
    X_scaled_full = StandardScaler().fit_transform(X)

    for model_name, base_model in models.items():
        print("\n" + "="*60)
        print(f"[MODEL] {model_name}")
        t_model_start = time.time()

        # 1) Randomized search (if defined)
        try:
            best_estimator, best_params = run_random_search(base_model, model_name, X_scaled_full, y, param_distributions)
        except Exception as e:
            print(f"[WARN] run_random_search threw exception for {model_name}: {e}. Using default wrapped estimator.")
            best_estimator = MultiOutputRegressor(clone(base_model), n_jobs=CFG.N_JOBS)
            best_params = {}

        # 2) Get CV predictions (manual kfold)
        try:
            cv_preds = manual_kfold_cv(best_estimator, X, y, n_splits=CFG.KFOLD, random_state=CFG.SEED)
        except Exception as e:
            print(f"[WARN] manual_kfold_cv failed for {model_name}: {e}. Filling zeros.")
            cv_preds = np.zeros_like(y)

        # 3) Train on full dataset with best estimator
        try:
            fitted_multi, X_scaled, y_pred_train = train_on_full(best_estimator, X, y)
        except Exception as e:
            print(f"[WARN] train_on_full failed for {model_name}: {e}")
            fitted_multi = clone(best_estimator)
            X_scaled = X_scaled_full
            y_pred_train = np.zeros_like(y)

        # 4) Compute metrics on CV predictions
        metrics = compute_metrics(y, cv_preds, target_names)
        print("[METRICS] " + pformat(metrics))

        # 5) Feature importance (on trained full model)
        try:
            importance_dict = get_feature_importance(fitted_multi, X_scaled, y, feature_cols, target_names)
            save_json_path = os.path.join(CFG.SAVE_DIR, f"{model_name}_feature_importance.json")
            save_json(importance_dict, save_json_path)
        except Exception as e:
            print(f"[WARN] get_feature_importance failed for {model_name}: {e}")
            importance_dict = {t: {f: 0.0 for f in feature_cols} for t in target_names}

        # 6) Save predictions, plots, high-error samples
        try:
            save_predictions(y, cv_preds, y_pred_train, model_name)
        except Exception as e:
            print(f"[WARN] save_predictions failed for {model_name}: {e}")

        try:
            plot_residuals(cv_preds, y, target_names, model_name)
            plot_pred_vs_true(y, y_pred_train, cv_preds, target_names, model_name)
            plot_feature_importance(importance_dict, feature_cols, target_names, model_name, topk=10)
            plot_high_error_samples(df, cv_preds, model_name)
        except Exception as e:
            print(f"[WARN] plotting failed for {model_name}: {e}")

        t_model_end = time.time()
        elapsed = t_model_end - t_model_start

        results_summary.append({
            "model": model_name,
            "metrics": metrics,
            "best_params": best_params,
            "elapsed_seconds": elapsed
        })

        print(f"[DONE] {model_name} completed in {elapsed:.1f}s")
   
    # Save summary
    try:
        save_json(results_summary, os.path.join(CFG.SAVE_DIR, "models_summary.json"))
    except Exception:
        pass

    print("\n[ALL DONE] Completed all models. Check 'results/' and 'plots/' directories.")
    return results_summary
# ---------------------------------------------------------------------
# 输出最优前9个模型
# ---------------------------------------------------------------------


import seaborn as sns



def plot_top9_cv_pred_vs_true_word(csv_path="binary_alloy_features_extended.csv",
                                   summary_json="results/models_summary.json",
                                   save_path="plots/top9_cv_pred_vs_true_word.png"):
    # Word友好字体
    plt.rcParams.update({
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16
    })

    sns.set_style("white")  # 干净背景

    # 读取数据
    df = pd.read_csv(csv_path)
    y_true = df[['E_far','E_nn']].values

    # 读取 summary
    with open(summary_json) as f:
        results_summary = json.load(f)

    # 按 E_far RMSE 排序前9个模型
    sorted_models = sorted(
        results_summary,
        key=lambda x: x['metrics'].get('E_far', {}).get('RMSE', np.inf)
    )[:9]

    # 创建 3x3 大图
    fig, axes = plt.subplots(3,3, figsize=(30,30))
    axes = axes.flatten()

    # 全局坐标范围
    y_min = y_true.min()
    y_max = y_true.max()

    for ax, res in zip(axes, sorted_models):
        model_name = res['model']
        pred_csv = os.path.join("results", f"{model_name}_predictions.csv")
        if not os.path.exists(pred_csv):
            print(f"[WARN] {pred_csv} not found, skipping {model_name}")
            continue
        pred_df = pd.read_csv(pred_csv)
        y_cv = pred_df[['E_far_cv_pred','E_nn_cv_pred']].values

        # 绘制散点
        ax.scatter(y_true[:,0], y_cv[:,0], color='blue', alpha=0.6, label='E_far')
        ax.scatter(y_true[:,1], y_cv[:,1], color='orange', alpha=0.6, label='E_nn')

        # y=x 参考线
        ax.plot([y_min, y_max],[y_min, y_max],'k--')

        # 计算 RMSE 和 R²
        rmse_far = np.sqrt(mean_squared_error(y_true[:,0], y_cv[:,0]))
        rmse_nn = np.sqrt(mean_squared_error(y_true[:,1], y_cv[:,1]))
        r2_far = r2_score(y_true[:,0], y_cv[:,0])
        r2_nn = r2_score(y_true[:,1], y_cv[:,1])

        # 设置标题
        ax.set_title(f"{model_name}\nRMSE_far={rmse_far:.2f}, R²_far={r2_far:.2f}\nRMSE_nn={rmse_nn:.2f}, R²_nn={r2_nn:.2f}")

        # 坐标轴标签带单位
        ax.set_xlabel("True (meV/Atoms)")
        ax.set_ylabel("Predicted (meV/Atoms)")

        # 正方形比例
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(y_min, y_max)
        ax.set_ylim(y_min, y_max)

        # 边框显示
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)

        # 坐标刻度向内
        ax.tick_params(direction='in', top=True, right=True)

        # 图例
        ax.legend(loc='upper left')

    plt.tight_layout(pad=4.0)  # 子图间距
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600)  # 高分辨率
    plt.close()
    print(f"[DONE] Top 9 CV prediction vs true (Word-friendly) saved to {save_path}")

def plot_top9_feature_importance_word(summary_json="results/models_summary.json",
                                      csv_path="binary_alloy_features_extended.csv",
                                      save_path="plots/top9_feature_importance_word.png",
                                      topk=10):
    # Word-friendly字体设置
    plt.rcParams.update({
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
        "legend.fontsize": 20
    })
    sns.set_style("white")  # 干净背景

    # 特征名映射为希腊字母/符号
    feature_labels = {
        'chi_A': r'$\chi_A$', 'chi_B': r'$\chi_B$', 'delta_chi': r'$\Delta \chi$',
        'r_A': r'$r_A$', 'r_B': r'$r_B$', 'delta_r': r'$\Delta r$',
        'Z_A': r'$Z_A$', 'Z_B': r'$Z_B$',
        'valence_A': r'$V_A$', 'valence_B': r'$V_B$', 'delta_valence': r'$\Delta V$',
        'V_atom_A': r'$V_\mathrm{atom}^A$', 'V_atom_B': r'$V_\mathrm{atom}^B$', 'delta_V': r'$\Delta V_\mathrm{atom}$',
        'EA_A': r'$EA_A$', 'EA_B': r'$EA_B$', 'delta_EA': r'$\Delta EA$',
        'IE_A': r'$IE_A$', 'IE_B': r'$IE_B$', 'delta_IE': r'$\Delta IE$',
        'chi_prod': r'$\chi_\mathrm{prod}$', 'chi_avg': r'$\chi_\mathrm{avg}$',
        'delta_chi_sq': r'$\Delta \chi^2$', 'delta_r_sq': r'$\Delta r^2$',
        'chi_r_interaction': r'$\chi r_\mathrm{int}$', 'delta_r_host_sum': r'$\Sigma \Delta r_\mathrm{host}$'
    }

    # 读取 summary
    with open(summary_json) as f:
        results_summary = json.load(f)

    # 按 E_far RMSE 排序前9个模型
    sorted_models = sorted(
        results_summary,
        key=lambda x: x['metrics'].get('E_far', {}).get('RMSE', np.inf)
    )[:9]

    # 创建 3x3 子图
    fig, axes = plt.subplots(3,3, figsize=(20,20))
    axes = axes.flatten()

    for ax, res in zip(axes, sorted_models):
        model_name = res['model']
        # 读取该模型的 feature importance
        fi_json = os.path.join("results", f"{model_name}_feature_importance.json")
        if not os.path.exists(fi_json):
            print(f"[WARN] {fi_json} not found, skipping {model_name}")
            continue
        with open(fi_json) as f:
            importance_dict = json.load(f)

        # 以 E_far 为例（第一目标）
        target_name = 'E_far'
        imp_map = importance_dict.get(target_name, {})
        if not imp_map:
            continue
        # 取 topk
        sorted_items = sorted(imp_map.items(), key=lambda x: x[1], reverse=True)[:topk]
        features, values = zip(*sorted_items)
        # 替换纵坐标为美观符号
        labels = [feature_labels.get(f, f) for f in features]

        ax.barh(labels, values, color='skyblue')
        ax.invert_yaxis()
        ax.set_title(f"{model_name} Top Features")
        ax.set_xlabel("Importance")
        ax.tick_params(direction='in', top=True, right=True)

    plt.tight_layout(pad=4.0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=600)
    plt.close()
    print(f"[DONE] Top 9 feature importance (Word-friendly) saved to {save_path}")




# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------
if __name__ == "__main__":

    #results_summary = run_full_pipeline()

    #plot_top9_cv_pred_vs_true_word()
    plot_top9_feature_importance_word()

