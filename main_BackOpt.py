#!/usr/bin/env python
import fire
from retry.api import retry_call
from tqdm import tqdm
from utils import NanException
from datetime import datetime
from run.TrainBackOptNet import Trainer


def train_from_folder(data='CustomNone',
                      results_dir='./GoodResult/results',
                      models_dir='./GoodResult/models',
                      log_dir='./GoodResult/logs',
                      name=f'BackOpt {datetime.now().hour}/{datetime.now().minute}',
                      new=False,
                      load_from_extract=2,
                      load_from_style=14,
                      batch_size=1,
                      num_train_steps=100,
                      learning_rate=2e-4,
                      save_every=100,
                      valid_acc=False,
                      StyleGAN_load_from=14):

    trainer = Trainer(name,
                      results_dir,
                      models_dir,
                      log_dir,
                      batch_size=batch_size,
                      lr=learning_rate,
                      save_every=save_every,)

    if not new:
        trainer.load_part_state_dict(load_from_style, load_from_extract)
    else:
        trainer.clear()

    # TODO
    if valid_acc:
        return

    train_now = datetime.now().timestamp()
    for _ in tqdm(range(num_train_steps - trainer.steps),
                  mininterval=10., desc=f'{name}<{data}>'):
        # train
        retry_call(trainer.train, tries=3, exceptions=NanException)

        # stop time
        if _ % 500 == 0:
            if datetime.now().timestamp() - train_now > 29880:
                break
        if _ % 10 == 0:
            pass
            # trainer.print_log()


if __name__ == "__main__":
    fire.Fire(train_from_folder)
