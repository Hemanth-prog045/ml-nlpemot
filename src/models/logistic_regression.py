import numpy as np
import pandas as pd
import time
import os
import psutil
import pickle
import logging
import warnings
from typing import Dict, Tuple, Optional, Union, List

from scipy.sparse import csr_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import (
    cross_validate,
    cross_val_predict,
    StratifiedKFold,
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

plt.style.use("default")
sns.set_palette("husl")

# ------------------------------------------------------------------
# PIPELINE
# ------------------------------------------------------------------
class EmotionClassifierPipeline:

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.cv_results = {}
        self.test_results = {}
        self.training_times = {}
        self.cv_times = {}
        self.resource_usage = {}

        logger.info(
            f"EmotionClassifierPipeline initialized "
            f"(scikit-learn {sklearn.__version__})"
        )

    # --------------------------------------------------------------
    # MODEL FACTORY
    # --------------------------------------------------------------
    def _get_model(self, model_type: str):
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=300,
                random_state=self.random_state,
                n_jobs=-1
            )

        elif model_type == "logistic_regression":
            return LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            )

        elif model_type == "svm":
            return SVC(
                kernel="rbf",
                probability=True,
                random_state=self.random_state
            )

        elif model_type == "naive_bayes":
            return MultinomialNB()

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    # --------------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------------
    def train_all_models(self, X_train, y_train, models):
        logger.info("=" * 80)
        logger.info("TRAINING ALL MODELS")
        logger.info("=" * 80)

        for model_type in models:
            logger.info(f"Training {model_type.replace('_',' ').title()}")

            model = self._get_model(model_type)

            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / (1024 ** 2)

            start = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start

            mem_after = process.memory_info().rss / (1024 ** 2)

            self.models[model_type] = model
            self.training_times[model_type] = train_time
            self.resource_usage[model_type] = {
                "memory_mb": mem_after - mem_before
            }

            logger.info(f"Training time: {train_time:.2f}s")

    # --------------------------------------------------------------
    # CROSS VALIDATION
    # --------------------------------------------------------------
    def cross_validate_all(self, X, y, cv=10):
        logger.info("=" * 80)
        logger.info("10-FOLD CROSS VALIDATION")
        logger.info("=" * 80)

        scoring = {
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted"
        }

        skf = StratifiedKFold(
            n_splits=cv,
            shuffle=True,
            random_state=self.random_state
        )

        for name, model in self.models.items():
            logger.info(f"CV for {name.replace('_',' ').title()}")

            start = time.time()
            cv_out = cross_validate(
                model,
                X,
                y,
                cv=skf,
                scoring=scoring,
                n_jobs=-1
            )
            cv_time = time.time() - start

            preds = cross_val_predict(
                model,
                X,
                y,
                cv=skf,
                n_jobs=-1
            )

            self.cv_results[name] = {
                "accuracy": {
                    "mean": cv_out["test_accuracy"].mean(),
                    "std": cv_out["test_accuracy"].std(),
                    "scores": cv_out["test_accuracy"]
                },
                "precision": {
                    "mean": cv_out["test_precision"].mean(),
                    "std": cv_out["test_precision"].std(),
                    "scores": cv_out["test_precision"]
                },
                "recall": {
                    "mean": cv_out["test_recall"].mean(),
                    "std": cv_out["test_recall"].std(),
                    "scores": cv_out["test_recall"]
                },
                "f1": {
                    "mean": cv_out["test_f1"].mean(),
                    "std": cv_out["test_f1"].std(),
                    "scores": cv_out["test_f1"]
                },
                "predictions": preds
            }

            self.cv_times[name] = cv_time

            logger.info(
                f"Accuracy: {self.cv_results[name]['accuracy']['mean']:.4f}"
            )

    # --------------------------------------------------------------
    # TEST EVALUATION
    # --------------------------------------------------------------
    def evaluate_all_models(self, X_test, y_test, label_names=None):
        logger.info("=" * 80)
        logger.info("TEST SET EVALUATION")
        logger.info("=" * 80)

        for name, model in self.models.items():
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="weighted")
            rec = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            try:
                y_proba = model.predict_proba(X_test)
                auc_score = roc_auc_score(
                    y_test, y_proba, multi_class="ovr"
                )
            except Exception:
                auc_score = None

            self.test_results[name] = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auc": auc_score,
                "confusion_matrix": confusion_matrix(y_test, y_pred),
                "report": classification_report(
                    y_test, y_pred, target_names=label_names
                )
            }

            logger.info(f"{name.title()} Accuracy: {acc:.4f}")

    # --------------------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------------------
    def get_comparison_table(self):
        rows = []

        for name in self.models:
            row = {"Model": name.replace("_"," ").title()}

            if name in self.cv_results:
                row["CV_Accuracy"] = (
                    f"{self.cv_results[name]['accuracy']['mean']:.4f}"
                )
                row["CV_F1"] = (
                    f"{self.cv_results[name]['f1']['mean']:.4f}"
                )

            if name in self.test_results:
                row["Test_Accuracy"] = (
                    f"{self.test_results[name]['accuracy']:.4f}"
                )
                row["Test_F1"] = (
                    f"{self.test_results[name]['f1']:.4f}"
                )

            rows.append(row)

        return pd.DataFrame(rows)

    # --------------------------------------------------------------
    # CV PLOT (FIXED GRID)
    # --------------------------------------------------------------
    def plot_cv_comparison(self, save_path=None):
        fig = plt.figure(figsize=(18,10))

        # FIXED: 3x4 grid for 4 metrics
        gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

        metrics = ["accuracy","precision","recall","f1"]
        colors = ["#3498db","#e74c3c","#2ecc71","#f39c12"]

        model_names = [m.replace("_"," ").title() for m in self.models]

        for idx, metric in enumerate(metrics):
            ax = fig.add_subplot(gs[0, idx])
            means = [self.cv_results[m][metric]["mean"] for m in self.models]
            stds = [self.cv_results[m][metric]["std"] for m in self.models]

            ax.bar(
                model_names,
                means,
                yerr=stds,
                capsize=5,
                color=colors[idx],
                alpha=0.7
            )
            ax.set_title(metric.upper())
            ax.set_ylim(0,1)
            ax.tick_params(axis="x", rotation=45)

        plt.suptitle(
            "Cross-Validation Performance Comparison",
            fontsize=16,
            fontweight="bold"
        )

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    # --------------------------------------------------------------
    # SAVE MODELS
    # --------------------------------------------------------------
    def save_all_models(self, directory="trained_models"):
        os.makedirs(directory, exist_ok=True)

        for name, model in self.models.items():
            with open(
                os.path.join(directory, f"{name}.pkl"), "wb"
            ) as f:
                pickle.dump(model, f)

            logger.info(f"Saved {name} model")


# ------------------------------------------------------------------
# DEMO RUN
# ------------------------------------------------------------------
if __name__ == "__main__":

    print("="*80)
    print("EMOTION CLASSIFICATION - FINAL FIXED VERSION")
    print(f"scikit-learn version: {sklearn.__version__}")
    print("="*80)

    np.random.seed(42)

    n_samples = 813
    n_features = 100
    n_classes = 9

    X = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)
    X = csr_matrix(X)

    labels = [
        "Positive","Trust","Fear","Negative","Anticipation",
        "Sadness","Anger","Surprise","Neutral"
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = EmotionClassifierPipeline(random_state=42)

    pipeline.train_all_models(
        X_train, y_train,
        ["random_forest","logistic_regression","svm","naive_bayes"]
    )

    pipeline.cross_validate_all(X_train, y_train)
    pipeline.evaluate_all_models(X_test, y_test, labels)

    df = pipeline.get_comparison_table()
    print("\nMODEL COMPARISON\n")
    print(df.to_string(index=False))

    pipeline.plot_cv_comparison(
        save_path="cv_comparison_fixed.png"
    )

    pipeline.save_all_models()

    print("\nPIPELINE EXECUTED SUCCESSFULLY")
