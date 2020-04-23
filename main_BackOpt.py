#!/usr/bin/env python
import fire
from retry.api import retry_call
from tqdm import tqdm
from run. import Trainer
from utils import NanException
from datetime import datetime


def train_from_folder(data='CustomNone',
                      results_dir='./GoodResult/results',
                      models_dir='./GoodResult/models',
                      log_dir='./GoodResult/logs',
                      name='ExtractNet',
                      new=False,
                      load_from=11,
                      batch_size=3,
                      num_train_steps=50000,
                      learning_rate=2e-4,
                      save_every=10000,
                      valid_acc=False,
                      StyleGAN_load_from=14):

    trainer = Trainer(name,
                      results_dir,
                      models_dir,
                      log_dir,
                      batch_size=batch_size,
                      lr=learning_rate,
                      save_every=save_every,)

    # init style gan not extract
    trainer.init_StyleGAN(StyleGAN_load_from)

    if not new:
        trainer.load_part_state_dict(load_from)
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
        if _ % 50 == 0:
            # TODO log
            pass


if __name__ == "__main__":
    fire.Fire(train_from_folder)
