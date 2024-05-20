'''
Create and register artifacts
Globant
2024-05-15
'''


# Description ---------------
# wandb.init(
#     project="",
#     group="",
#     job_type="",
#     config={}
# )

# Libraries -----------------

import wandb


# Create Artifact ---------------

with open('one_artifact.txt', 'w+') as fp:
    fp.write('Green is a color!!')


# Upload to Weights and Bias ---------

run = wandb.init(project='artifact_example',
                 group='random_examples')



# Create artifact in W&D ------

artifact = wandb.Artifact(
    name='example.txt',
    type='data',
    description='example of an artifact',
    metadata={'key': 'example_1'}
)

# Adding Artifact to track -------
artifact.add_file('one_artifact.txt')

run.log_artifact(artifact)

run.finish()

