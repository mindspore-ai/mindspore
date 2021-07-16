# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" STGAN TRAIN"""
import tqdm

from mindspore.common import set_seed

from src.models import STGANModel
from src.utils import get_args
from src.dataset import CelebADataLoader

set_seed(1)

def train():
    """Train Function"""
    args = get_args("train")
    print(args)

    print('\n\n=============== start training ===============\n\n')

    # Get DataLoader
    data_loader = CelebADataLoader(args.dataroot,
                                   mode=args.phase,
                                   selected_attrs=args.attrs,
                                   batch_size=args.batch_size,
                                   image_size=args.image_size,
                                   device_num=args.device_num)
    iter_per_epoch = len(data_loader)
    args.dataset_size = iter_per_epoch

    # Get STGAN MODEL
    model = STGANModel(args)
    it_count = 0

    for _ in tqdm.trange(args.n_epochs, desc='Epoch Loop'):
        for _ in tqdm.trange(iter_per_epoch, desc='Inner Epoch Loop'):
            if model.current_iteration > it_count:
                it_count += 1
                continue
            try:
                # training model
                data = next(data_loader.train_loader)
                model.set_input(data)
                model.optimize_parameters()

                # saving model
                if (it_count + 1) % args.save_freq == 0:
                    model.save_networks()

                # sampling
                if (it_count + 1) % args.sample_freq == 0:
                    model.eval(data_loader)

            except KeyboardInterrupt:
                logger.info('You have entered CTRL+C.. Wait to finalize')
                model.save_networks()

            it_count += 1
            model.current_iteration = it_count

    model.save_networks()
    print('\n\n=============== finish training ===============\n\n')


if __name__ == '__main__':
    train()
