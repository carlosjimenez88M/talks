'''
Introduction
Globant
2024-05-17
'''


# Description --------------------
# https://docs.wandb.ai/quickstart

# wandb.init(
#     project="",
#     group="",
#     job_type="",
#     config={}
# )

# Libraries ----------------------

import argparse
import logging
import random
import wandb


## Logger Configuration ----------

logging.basicConfig(
	filename='./example.log',
	level = logging.INFO,
	filemode = 'w',
	format = '%(name)s - %(levelname)s - %(message)s'
	)

logger = logging.getLogger()

# Create Artifact -----


def go():
    run =  wandb.init(
        project='example_1',
        config={
            'learning_rate': args.learning_rate,
            'epochs': args.epochs
        }
    )

    logger.info('Initialize example......')

    offset=random.random()/8
    logger.info(f'offset: {offset}')
    logger.info(f"lr: {args.learning_rate}")

    for epoch in range(2, args.epochs):
        acc = 1 - 2 ** -epoch - random.random() / epoch - offset
        loss = 2 ** -epoch + random.random() / epoch + offset
        logging.info(f"epoch={epoch}, accuracy={acc}, loss={loss}")
        print(f"epoch={epoch}, accuracy={acc}, loss={loss}")
        wandb.log({"accuracy": acc, "loss": loss})

    run.log_code()

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='WandB Example'
    )
    parser.add_argument('--project',
                        type=str,
                        default='Globant',
                        help='WandB project name')
    parser.add_argument('--epochs',
                        type=int,
                        default=10,
                        help='Number of epochs')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.01,
                        help='Learning rate')
    args = parser.parse_args()

    go()


