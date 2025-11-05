import logging

from steps.ingest import Ingestion
from steps.clean import Cleaner
from steps.train import Trainer
from steps.predict import Predictor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


def main():
    # Data ingestion
    ingestion = Ingestion(config_path="config.yml")
    train, test = ingestion.load_data()
    logging.info("Data ingestion completed")

    # Data cleansing
    cleaner = Cleaner()
    train_data = cleaner.clean_data(train)
    test_data = cleaner.clean_data(test)
    logging.info("Data cleaning completed")

    # Prepare and Train Model
    trainer = Trainer()
    X_train, y_train = trainer.feature_target_separator(train_data)
    trainer.train_model(X_train, y_train)
    trainer.save_model()
    logging.info("Model training completed")

    # Evaluate Model
    predictor = Predictor()
    X_test, y_test = predictor.feature_target_separator(test_data)
    accuracy, class_report, roc_auc = predictor.eval_model(X_test, y_test)
    logging.info("Model evaluation completed")

    print(f"Model: {trainer.model_name}")
    print(f"Accuracy score: {accuracy: 4f}, ROC AUC Score: {roc_auc: .4f}")
    print(f"\n{class_report}")


def train_with_mlflow():
    ...


if __name__ == "__main__":
    main()
    # train_with_mlflow()