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
"""
Visualization for testing purposes.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np

import mindspore.dataset as ds
from mindspore import context
context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


def compare(data_path, index=None, ops=None, rescale=False):
    """Visualize images before and after applying auto-augment."""
    # Load dataset
    ds.config.set_seed(8)
    dataset_orig = ds.Cifar10Dataset(
        data_path,
        num_samples=5,
        shuffle=True,
    )

    # Apply transformations
    dataset_augmented = dataset_orig.map(
        operations=[Augment(index)] if ops is None else ops,
        input_columns=['image'],
    )

    # Collect images
    image_orig_list, image_augmented_list, label_list = [], [], []
    for data in dataset_orig.create_dict_iterator():
        image_orig_list.append(data['image'])
        label_list.append(data['label'])
        print('Original image:  shape {}, label {}'.format(
            data['image'].shape, data['label'],
        ))
    for data in dataset_augmented.create_dict_iterator():
        image_augmented_list.append(data['image'])
        print('Augmented image: shape {}, label {}'.format(
            data['image'].shape, data['label'],
        ))

    num_samples = len(image_orig_list)
    fig, mesh = plt.subplots(ncols=num_samples, nrows=2, figsize=(5, 3))
    axes = mesh[0]
    for i in range(num_samples):
        axes[i].axis('off')
        axes[i].imshow(image_orig_list[i].asnumpy())
        axes[i].set_title(label_list[i].asnumpy())
    axes = mesh[1]
    for i in range(num_samples):
        axes[i].axis('off')
        img = image_augmented_list[i].asnumpy().transpose((1, 2, 0))
        if rescale:
            max_val = max(np.abs(img.min()), img.max())
            img = (img / max_val + 1) / 2
        print('min and max of the transformed image:', img.min(), img.max())
        axes[i].imshow(img)
    fig.tight_layout()
    fig.savefig(
        'aug_test.png' if index is None else 'aug_test_{}.png'.format(index),
    )


if __name__ == '__main__':
    sys.path.append('..')
    from autoaugment.third_party.policies import good_policies
    from autoaugment import Augment

    cifar10_data_path = './cifar-10-batches-bin/'

    # Test the feasibility of each policy
    for ind, policy in enumerate(good_policies()):
        if ind >= 3:
            pass
            # break
        print(policy)
        compare(cifar10_data_path, ind)

    # Test the random policy selection and the normalize operation
    MEAN = [0.49139968, 0.48215841, 0.44653091]
    STD = [0.24703223, 0.24348513, 0.26158784]
    compare(
        cifar10_data_path,
        ops=[Augment(mean=MEAN, std=STD, enable_basic=False)],
    )
    compare(
        cifar10_data_path,
        ops=[Augment(mean=MEAN, std=STD, enable_basic=False)],
        rescale=True,
    )
