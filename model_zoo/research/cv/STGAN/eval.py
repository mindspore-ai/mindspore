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
""" Model Test """
import tqdm

from mindspore.common import set_seed

from src.models import STGANModel
from src.utils import get_args
from src.dataset import CelebADataLoader

set_seed(1)

def test():
    """ test function """
    args = get_args("test")
    print('\n\n=============== start testing ===============\n\n')
    data_loader = CelebADataLoader(args.dataroot,
                                   mode=args.phase,
                                   selected_attrs=args.attrs,
                                   batch_size=1,
                                   image_size=args.image_size)
    iter_per_epoch = len(data_loader)
    args.dataset_size = iter_per_epoch
    model = STGANModel(args)

    for _ in tqdm.trange(iter_per_epoch, desc='Test Loop'):
        data = next(data_loader.test_loader)
        model.test(data, data_loader.test_set.get_current_filename())

    print('\n\n=============== finish testing ===============\n\n')


if __name__ == '__main__':
    test()
