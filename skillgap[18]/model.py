# model.py
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, silhouette_score, calinski_harabasz_score, davies_bouldin_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Optional: install xgboost if not already
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class SkillGapModel:
    """
    Optimized Skill Gap Detection & Employability Prediction
    - PCA for dimensionality reduction
    - KMeans clustering for skill profiling
    - Skill gap computation
    - Supervised employability prediction (Random Forest / XGBoost)
    - Multi-model evaluation with cross-validation
    """

    SKILLS = [
    "programming_fundamentals",
    "data_structures",
    "web_development",
    "software_engineering",
    "system_design",
    "machine_learning",
    "data_visualization",
    "databases",
    "operating_systems",
    "computer_networks",
    "cloud_computing",
    "devops"
    ]

    def __init__(self, n_components=0.95, n_clusters=4):
        """n_components: PCA variance to retain (float <1.0) or int components
           n_clusters: Number of KMeans clusters"""
        self.n_clusters = n_clusters
        self.n_components = n_components

        self.scaler = StandardScaler()
        self.pca = None
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)

        self.df = None
        self.skill_cols = list(self.SKILLS)

        # Results
        self.X_scaled = None
        self.X_pca = None
        self.labels = None

        # Clustering metrics
        self.silhouette = None
        self.calinski_harabasz = None
        self.davies_bouldin = None

    # ---------------- DATA ----------------
    def load_dataframe(self, df: pd.DataFrame):
        df = df.copy().reset_index(drop=True)
        df = self.aggregate_raw_skills(df)

        self.skill_cols = self.SKILLS
        self.df = df

        return self.df

    def preprocess(self, use_pca=True):
        self.X_scaled = self.scaler.fit_transform(self.df[self.skill_cols].values)
        if use_pca:
            self.pca = PCA(n_components=self.n_components, random_state=42)
            self.X_pca = self.pca.fit_transform(self.X_scaled)
        else:
            self.X_pca = self.X_scaled
        return self.X_pca

    # ---------------- KMEANS ----------------
    def run_kmeans(self):
        self.labels = self.kmeans.fit_predict(self.X_scaled)
        self.df["cluster"] = self.labels

        if len(set(self.labels)) > 1:
            self.silhouette = silhouette_score(self.X_scaled, self.labels)
            self.calinski_harabasz = calinski_harabasz_score(self.X_scaled, self.labels)
            self.davies_bouldin = davies_bouldin_score(self.X_scaled, self.labels)
        return self.labels

    # ---------------- SKILL GAP ----------------
    def compute_skill_gaps(self, ideal_target=8.0):
        cluster_means = self.df.groupby("cluster")[self.skill_cols].mean()
        worst_cluster = cluster_means.min(axis=0)
        overall_mean = self.df[self.skill_cols].mean()

        gap_df = pd.DataFrame({
            "overall_mean": overall_mean,
            "worst_cluster_mean": worst_cluster,
            "gap": ideal_target - worst_cluster
        }).sort_values("gap", ascending=False)

        return gap_df

    # ---------------- SUPERVISED LABEL ----------------
    def _create_employability_labels(self, threshold=7.0):
        avg_skill = self.df[self.skill_cols].mean(axis=1)
        return (avg_skill >= threshold).astype(int)

    # ---------------- SINGLE MODEL EVALUATION ----------------
    def supervised_accuracy(self, model=None, use_pca=True, threshold=7.0):
        if model is None:
            model = RandomForestClassifier(n_estimators=500, min_samples_split=5, random_state=42)

        X = self.preprocess(use_pca=use_pca)
        y = self._create_employability_labels(threshold)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1-score": f1_score(y_test, y_pred)
        }

    def aggregate_raw_skills(self, df: pd.DataFrame):
        """
        Safely converts raw skill columns into grouped competencies
        """
        grouped = pd.DataFrame(index=df.index)

        # ---------- Programming ----------
        prog_cols = [c for c in ["python", "java", "c++", "javascript"] if c in df.columns]
        grouped["programming_fundamentals"] = df[prog_cols].mean(axis=1) if prog_cols else 0

        # ---------- Data Structures ----------
        grouped["data_structures"] = df["data_structures"] if "data_structures" in df.columns else 0

        # ---------- Web Development ----------
        if "web_development" in df.columns:
            grouped["web_development"] = df["web_development"]
        else:
            web_cols = [c for c in ["html", "css"] if c in df.columns]
            grouped["web_development"] = df[web_cols].mean(axis=1) if web_cols else 0

        # ---------- Software Engineering ----------
        grouped["software_engineering"] = df["software_engineering"] if "software_engineering" in df.columns else 0

        # ---------- System Design ----------
        grouped["system_design"] = df["system_design"] if "system_design" in df.columns else 0

        # ---------- Machine Learning ----------
        ml_cols = [c for c in ["machine_learning", "deep_learning", "ai_ml_engineering"] if c in df.columns]
        grouped["machine_learning"] = df[ml_cols].mean(axis=1) if ml_cols else 0

        # ---------- Data Visualization ----------
        grouped["data_visualization"] = df["data_visualization"] if "data_visualization" in df.columns else 0

        # ---------- Databases ----------
        db_cols = [c for c in ["sql", "mongodb", "dbms"] if c in df.columns]
        grouped["databases"] = df[db_cols].mean(axis=1) if db_cols else 0

        # ---------- Systems ----------
        grouped["operating_systems"] = df["operating_systems"] if "operating_systems" in df.columns else 0
        grouped["computer_networks"] = df["computer_networks"] if "computer_networks" in df.columns else 0
        grouped["cloud_computing"] = df["cloud_computing"] if "cloud_computing" in df.columns else 0

        # ---------- DevOps ----------
        devops_cols = [c for c in ["git_github", "docker", "kubernetes", "devops"] if c in df.columns]
        grouped["devops"] = df[devops_cols].mean(axis=1) if devops_cols else 0

        return grouped.fillna(0)


    # ---------------- MULTI-MODEL CROSS-VALIDATION ----------------
    def evaluate_models(self, threshold=7.0, use_pca=True):
        y = self._create_employability_labels(threshold)
        X = self.preprocess(use_pca=use_pca)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=500, min_samples_split=5, random_state=42),
            "SVM": SVC(kernel="rbf", probability=True, random_state=42)
        }

        if XGBClassifier:
            models["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=5, use_label_encoder=False, eval_metric='logloss', random_state=42)

        results = []

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        for name, model in models.items():
            accuracy = cross_val_score(model, X, y, cv=skf, scoring="accuracy").mean()
            precision = cross_val_score(model, X, y, cv=skf, scoring="precision").mean()
            recall = cross_val_score(model, X, y, cv=skf, scoring="recall").mean()
            f1 = cross_val_score(model, X, y, cv=skf, scoring="f1").mean()

            results.append({
                "Model": name,
                "Accuracy": round(accuracy, 3),
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "F1-score": round(f1, 3)
            })

        return pd.DataFrame(results)
