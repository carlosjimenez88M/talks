'''
Download dataset with Wand and MLflow
Step1
Globant
2024-05-17
'''


# Libraries ----------------------

import argparse
import logging
import pathlib

import requests
import tempfile
import wandb


## Logger Configuration ----------

logging.basicConfig(
	filename='../logs/download_data.log',
	level= logging.INFO,
	filemode= 'w',
	format= '%(name)s - %(levelname)s - %(message)s')


logger = logging.getLogger()

## Main function ---------


def go(args):
	# pathlib.Path(args.file_url) = Convert url in object
	# name = access bane
	# .split("?")[0] = split name
	# .split("#")[0] = split name
	basename = pathlib.Path(args.file_url).name.split("?")[0].split('#')[0]

	logger.info(f"Downloading file from {args.file_url}")
	with tempfile.NamedTemporaryFile(mode='wb+', delete=False) as temp_fp:
		logger.info('Creating run')
		with wandb.init(
				project='Pokemon_exercise',
				job_type='download_data'
		) as run:
			with requests.get(args.file_url, stream=True) as r:
				for chunk in r.iter_content(chunk_size=8192):
					temp_fp.write(chunk)
			temp_fp.flush()

			logger.info('Creating artifact')
			artifact = wandb.Artifact(
				name=args.artifact_name,
				type=args.artifact_type,
				description=args.artifact_description,
				metadata={'original_url': args.file_url}
			)
			artifact.add_file(temp_fp.name, name=basename)
			run.log_artifact(artifact)

			artifact.wait()


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description='Pokemon project'
	)
	parser.add_argument(
		"--file_url",
		type=str,
		help="URL to the input file",
		required=True
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




