#!/usr/bin/env python
import fire
from retry.api import retry_call
from tqdm import tqdm
from run.TrainStylegan2WithNoise import Trainer
from utils import NanException
from datetime import datetime


def train_from_folder(data='../../gan/custom_dataset',
                      results_dir='./GoodResult/results',
                      models_dir='./GoodResult/models',
                      log_dir='./GoodResult/logs',
                      name='mytest',
                      new=False,
                      load_from=10,
                      image_size=64,
                      network_capacity=16,
                      transparent=False,
                      batch_size=3,
                      gradient_accumulate_every=5,
                      num_train_steps=120000,
                      learning_rate=2e-4,
                      num_workers=None,
                      save_every=10000,
                      generate=False,
                      num_image_tiles=8,
                      trunc_psi=0.6):
                      
    trainer = Trainer(name,
                    results_dir,
                    models_dir,
                    log_dir,
                    batch_size=batch_size,
                    gradient_accumulate_every=gradient_accumulate_every,
                    image_size=image_size,
                    network_capacity=network_capacity,
                    transparent=transparent,
                    lr=learning_rate,
                    num_workers=num_workers,
                    save_every=save_every,
                    trunc_psi=trunc_psi)

    if not new:
        trainer.load_part_state_dict(load_from)
    else:
        trainer.clear()

    if generate:
        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
        samples_name = f'generated-{timestamp}'
        trainer.evaluate(samples_name, num_image_tiles)
        print(
            f'sample images generated at {results_dir}/{name}/{samples_name}')
        return

    trainer.set_data_src(data)

    train_now = datetime.now().timestamp()
    for _ in tqdm(range(num_train_steps - trainer.steps),
                  mininterval=10., desc=f'{name}<{data}>'):
        # train
        retry_call(trainer.train, tries=3, exceptions=NanException)

        # stop time
        if _ % 500 == 0:
            if datetime.now().timestamp() - train_now > 29880:
                break
        if _ % 50 == 0:
            trainer.print_log()


if __name__ == "__main__":
    fire.Fire(train_from_folder)
