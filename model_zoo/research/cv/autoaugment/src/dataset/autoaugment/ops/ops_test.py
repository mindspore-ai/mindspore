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

import matplotlib.pyplot as plt

import mindspore.dataset as ds
import mindspore.dataset.vision.py_transforms as py_trans
from mindspore import context
context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')


def compare(data_path, trans, output_path='./ops_test.png'):
    """Visualize images before and after applying the given transformations."""
    # Load dataset
    ds.config.set_seed(8)
    dataset_orig = ds.Cifar10Dataset(
        data_path,
        num_samples=5,
        shuffle=True,
    )

    # Apply transformations
    dataset_augmented = dataset_orig.map(
        operations=[py_trans.ToPIL()] + trans + [py_trans.ToTensor()],
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

    # Plot images
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
        axes[i].imshow(image_augmented_list[i].asnumpy().transpose((1, 2, 0)))
    fig.tight_layout()
    fig.savefig(output_path)


if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from ops import OperatorClasses
    oc = OperatorClasses()

    levels = {
        'Contrast': 7,
        'Rotate': 7,
        'TranslateX': 9,
        'Sharpness': 9,
        'ShearY': 8,
        'TranslateY': 9,
        'AutoContrast': 9,
        'Equalize': 9,
        'Solarize': 8,
        'Color': 9,
        'Posterize': 7,
        'Brightness': 9,
        'Cutout': 4,
        'ShearX': 4,
        'Invert': 7,
        'RandomHorizontalFlip': None,
        'RandomCrop': None,
    }

    cifar10_data_path = './cifar-10-batches-bin/'
    for name, op in oc.vars().items():
        compare(cifar10_data_path, [op(levels[name])], './ops_{}_{}.png'.format(
            name, levels[name],
        ))
