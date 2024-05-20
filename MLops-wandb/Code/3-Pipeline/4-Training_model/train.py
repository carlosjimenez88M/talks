import argparse
import logging
import pandas as pd
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

## Logger Configuration -------
logging.basicConfig(
    filename='../logs/training.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger()

def main():
    parser = argparse.ArgumentParser(description="Train a RandomForest model")
    parser.add_argument("--train_artifact",
                        type=str,
                        help="Path to the training data artifact",
                        required=True)
    parser.add_argument("--test_artifact",
                        type=str,
                        help="Path to the test data artifact",
                        required=True)
    args = parser.parse_args()

    run = wandb.init(project="Pokemon_exercise",
                     job_type="train_model")

    logger.info("Downloading train artifact")
    train_artifact = run.use_artifact(args.train_artifact)
    train_path = train_artifact.file()

    logger.info("Downloading test artifact")
    test_artifact = run.use_artifact(args.test_artifact)
    test_path = test_artifact.file()

    logger.info("Reading train data")
    train_df = pd.read_csv(train_path)
    logger.info("Reading test data")
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["total"])
    y_train = train_df["total"]

    X_test = test_df.drop(columns=["total"])
    y_test = test_df["total"]

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    logger.info("Training model")
    model.fit(X_train, y_train)

    logger.info("Evaluating model")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    logger.info(f"Mean Squared Error: {mse}")
    logger.info(f"R^2 Score: {r2}")
    logger.info(f"Mean Absolute Error: {mae}")

    wandb.log({
        "mse": mse,
        "r2": r2,
        "mae": mae
    })

    logger.info("Saving model")
    model_path = "random_forest_model.pkl"
    pd.to_pickle(model, model_path)

    artifact = wandb.Artifact(
        name="random_forest_model",
        type="model",
        description="Random Forest model trained on Pokemon data",
    )
    artifact.add_file(model_path)

    logger.info("Logging model artifact")
    run.log_artifact(artifact)

if __name__ == "__main__":
    main()
