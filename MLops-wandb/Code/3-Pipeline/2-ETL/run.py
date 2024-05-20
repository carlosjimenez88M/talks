'''
ETL
Step2
GLobant
2024-05-17
'''


# Libraries -----------
import argparse
import logging
import pandas as pd
import wandb
from sklearn.preprocessing import OneHotEncoder

## Logger Configuration -------
logging.basicConfig(
    filename='../logs/etl.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger()

## Main function ----------
def go(args):
    run = wandb.init(project='Pokemon_exercise',
                     job_type="process_data")

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    df = pd.read_csv(artifact_path)


    new_columns = ['#','name', 'type_1', 'type_2', 'total', 'hp', 'attack', 'defense',
                   'sp_atk', 'sp_def', 'speed', 'generation', 'legendary']
    df.columns = new_columns

    logger.info('Remove NaN values')
    df = df.dropna().drop(columns=['name']).drop_duplicates().reset_index(drop=True)

    logger.info('One Hot Encoder transformation')
    categorical_columns = ['type_1', 'type_2', 'generation', 'legendary']

    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_features = encoder.fit_transform(df[categorical_columns])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_columns))
    df = pd.concat([df.drop(categorical_columns, axis=1), encoded_df], axis=1)
    print(df.head())

    logger.info("Creating artifact")
    df.to_csv("clean_data.csv", index=False)

    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    logger.info('Creating artifact')
    artifact.add_file("clean_data.csv")

    logger.info("Logging artifact")
    run.log_artifact(artifact)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a file artifact to W&B",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name",
        type=str,
        help="Name for the artifact",
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

    args = parser.parse_args()
    go(args)
