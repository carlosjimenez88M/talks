'''
Split dataset
Globant
2024-05-17
'''

# Libraries ---------
import argparse
import logging
import pandas as pd
import wandb
from sklearn.model_selection import train_test_split

## Logger Configuration -------
logging.basicConfig(
    filename='../logs/split.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger()

## Main function ----------
def go(args):
    run = wandb.init(
        job_type="split_data"
    )

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)

    logger.info('Splitting data into train and test sets')
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=42)

    logger.info("Saving train and test data")
    train_df.to_csv("train_data.csv", index=False)
    test_df.to_csv("test_data.csv", index=False)

    logger.info("Creating train data artifact")
    train_artifact = wandb.Artifact(
        name=args.train_artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    train_artifact.add_file("train_data.csv")

    logger.info("Logging train data artifact")
    run.log_artifact(train_artifact)

    logger.info("Creating test data artifact")
    test_artifact = wandb.Artifact(
        name=args.test_artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    test_artifact.add_file("test_data.csv")

    logger.info("Logging test data artifact")
    run.log_artifact(test_artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split data into train and test sets",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--train_artifact_name",
        type=str,
        help="Name for the train artifact",
        required=True
    )

    parser.add_argument(
        "--test_artifact_name",
        type=str,
        help="Name for the test artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_type",
        type=str,
        help="Type for the artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    parser.add_argument(
        "--test_size",
        type=float,
        help="Proportion of data to use for the test set",
        required=True,
    )

    args = parser.parse_args()

    go(args)
