import mlflow
import os
import wandb
import hydra
from omegaconf import DictConfig

@hydra.main(config_name='config')
def go(config: DictConfig):
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    root_path = hydra.utils.get_original_cwd()

    # Descargar datos
    _ = mlflow.run(
        os.path.join(root_path, "1-Download_data"),
        "main",
        parameters={
            "file_url": config["data"]["file_url"],
            "artifact_name": "pokemon.csv",
            "artifact_type": "raw_data",
            "artifact_description": "Input data"
        },
    )


    _ = mlflow.run(
        os.path.join(root_path, "2-ETL"),
        "main",
        parameters={
            "input_artifact": "pokemon.csv:latest",
            "artifact_name": "clean_data.csv",
            "artifact_type": "processed_data",
            "artifact_description": "Cleaned data"
        },
    )


    _ = mlflow.run(
        os.path.join(root_path, "3-Split_dataset"),
        "main",
        parameters={
            "input_artifact": "clean_data.csv:latest",
            "train_artifact_name": "train_data.csv",
            "test_artifact_name": "test_data.csv",
            "artifact_type": "split_data",
            "artifact_description": "Train and test split",
            "test_size": 0.2
        },
    )


    _ = mlflow.run(
        os.path.join(root_path, "4-Training_model"),
        "main",
        parameters={
            "train_artifact": "train_data.csv:latest",
            "test_artifact": "test_data.csv:latest"
        },
    )

if __name__ == "__main__":
    go()
