import os
import sys
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)

# ============================
# MLflow / DagsHub Configuration
# ============================
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Rajamohannaidu/networksecurity.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "Rajamohannaidu"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "22e234815e0be32f1a5056ba723c657f804b221d"


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # ============================
    # MLflow Tracking
    # ============================
    def track_mlflow(self, best_model, train_metric, test_metric):
        try:
            mlflow.set_registry_uri(
                "https://dagshub.com/Rajamohannaidu/networksecurity.mlflow"
            )

            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            model_name = best_model.__class__.__name__

            with mlflow.start_run():
                # Train metrics
                mlflow.log_metric("train_f1", train_metric.f1_score)
                mlflow.log_metric("train_precision", train_metric.precision_score)
                mlflow.log_metric("train_recall", train_metric.recall_score)

                # Test metrics
                mlflow.log_metric("test_f1", test_metric.f1_score)
                mlflow.log_metric("test_precision", test_metric.precision_score)
                mlflow.log_metric("test_recall", test_metric.recall_score)

                # Log & register model
                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(
                        sk_model=best_model,
                        artifact_path="model",
                        registered_model_name=model_name,
                    )
                else:
                    mlflow.sklearn.log_model(best_model, "model")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # ============================
    # Model Training
    # ============================
    def train_model(self, X_train, y_train, X_test, y_test):
        try:
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "AdaBoost": AdaBoostClassifier(),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"]
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 128, 256]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Logistic Regression": {},
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]
            logging.info(f"Best model selected: {best_model_name}")

            # Metrics
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_metric = get_classification_score(y_train, y_train_pred)
            test_metric = get_classification_score(y_test, y_test_pred)

            # MLflow Tracking
            self.track_mlflow(best_model, train_metric, test_metric)

            # Load preprocessor
            preprocessor = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            # Save final model
            model_dir = os.path.dirname(
                self.model_trainer_config.trained_model_file_path
            )
            os.makedirs(model_dir, exist_ok=True)

            network_model = NetworkModel(
                preprocessor=preprocessor, model=best_model
            )

            save_object(
                self.model_trainer_config.trained_model_file_path,
                network_model,
            )

            # Optional standalone model save
            save_object("final_model/model.pkl", best_model)

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric,
            )

            logging.info(f"Model trainer artifact created: {model_trainer_artifact}")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    # ============================
    # Pipeline Entry
    # ============================
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
