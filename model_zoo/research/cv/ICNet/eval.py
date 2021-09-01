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
"""Evaluate mIou and Pixacc"""
import os
import time
import sys
import argparse
import yaml
import numpy as np
from PIL import Image
import mindspore.ops as ops
from mindspore import load_param_into_net
from mindspore import load_checkpoint
from mindspore import Tensor
import mindspore.dataset.vision.py_transforms as transforms

parser = argparse.ArgumentParser(description="ICNet Evaluation")
parser.add_argument("--dataset_path", type=str, default="/data/cityscapes/", help="dataset path")
parser.add_argument("--checkpoint_path", type=str, default="/root/ICNet/ckpt/ICNet-160_93_699.ckpt",
                    help="checkpoint_path, default67.7")
parser.add_argument("--project_path", type=str, default='/root/ICNet/',
                    help="project_path,default is /root/ICNet/")
parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")

args_opt = parser.parse_args()


class Evaluator:
    """evaluate"""

    def __init__(self, config):
        self.cfg = config

        # get valid dataset images and targets
        self.image_paths, self.mask_paths = _get_city_pairs(config["train"]["cityscapes_root"], "val")

        # create network
        self.model = ICNet(nclass=19, pretrained_path=cfg["train"]["pretrained_model_path"], istraining=False)

        # load ckpt
        ckpt_file_name = args_opt.checkpoint_path
        param_dict = load_checkpoint(ckpt_file_name)
        load_param_into_net(self.model, param_dict)

        # evaluation metrics
        self.metric = SegmentationMetric(19)

    def eval(self):
        """evaluate"""
        self.metric.reset()
        model = self.model
        model = model.set_train(False)

        logger.info("Start validation, Total sample: {:d}".format(len(self.image_paths)))
        list_time = []

        for i in range(len(self.image_paths)):
            image = Image.open(self.image_paths[i]).convert('RGB')  # image shape: (W,H,3)
            mask = Image.open(self.mask_paths[i])  # mask shape: (W,H)

            image = self._img_transform(image)  # image shape: (3,H,W) [0,1]
            mask = self._mask_transform(mask)  # mask shape: (H,w)

            image = Tensor(image)

            expand_dims = ops.ExpandDims()
            image = expand_dims(image, 0)

            start_time = time.time()
            output = model(image)
            end_time = time.time()
            step_time = end_time - start_time

            output = output.asnumpy()
            mask = np.expand_dims(mask.asnumpy(), axis=0)
            self.metric.update(output, mask)
            list_time.append(step_time)

        mIoU, pixAcc = self.metric.get()

        average_time = sum(list_time) / len(list_time)

        print("avgmiou", mIoU)
        print("avg_pixacc", pixAcc)
        print("avgtime", average_time)

    def _img_transform(self, image):
        """img_transform"""
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        image = to_tensor(image)
        image = normalize(image)
        return image

    def _mask_transform(self, mask):
        mask = self._class_to_index(np.array(mask).astype('int32'))
        return Tensor(np.array(mask).astype('int32'))  # torch.LongTensor

    def _class_to_index(self, mask):
        """assert the value"""
        values = np.unique(mask)
        self._key = np.array([-1, -1, -1, -1, -1, -1,
                              -1, -1, 0, 1, -1, -1,
                              2, 3, 4, -1, -1, -1,
                              5, -1, 6, 7, 8, 9,
                              10, 11, 12, 13, 14, 15,
                              -1, -1, 16, 17, 18])
        self._mapping = np.array(range(-1, len(self._key) - 1)).astype('int32')
        for value in values:
            assert value in self._mapping
        # Get the index of each pixel value in the mask corresponding to _mapping
        index = np.digitize(mask.ravel(), self._mapping, right=True)
        # According to the above index index, according to _key, the corresponding mask image is obtained
        return self._key[index].reshape(mask.shape)


def _get_city_pairs(folder, split='train'):
    """get dataset img_mask_path_pairs"""

    def get_path_pairs(image_folder, mask_folder):
        img_paths = []
        mask_paths = []
        for root, _, files in os.walk(image_folder):
            for filename in files:
                if filename.endswith('.png'):
                    imgpath = os.path.join(root, filename)
                    foldername = os.path.basename(os.path.dirname(imgpath))
                    maskname = filename.replace('leftImg8bit', 'gtFine_labelIds')
                    maskpath = os.path.join(mask_folder, foldername, maskname)
                    if os.path.isfile(imgpath) and os.path.isfile(maskpath):
                        img_paths.append(imgpath)
                        mask_paths.append(maskpath)
                    else:
                        print('cannot find the mask or image:', imgpath, maskpath)
        print('Found {} images in the folder {}'.format(len(img_paths), image_folder))
        return img_paths, mask_paths

    if split in ('train', 'val', 'test'):
        # "./Cityscapes/leftImg8bit/train" or "./Cityscapes/leftImg8bit/val"
        img_folder = os.path.join(folder, 'leftImg8bit/' + split)
        # "./Cityscapes/gtFine/train" or "./Cityscapes/gtFine/val"
        mask_folder = os.path.join(folder, 'gtFine/' + split)

        img_paths, mask_paths = get_path_pairs(img_folder, mask_folder)
        return img_paths, mask_paths


if __name__ == '__main__':
    sys.path.append(args_opt.project_path)
    from src.models import ICNet
    from src.metric import SegmentationMetric
    from src.logger import SetupLogger
    # Set config file
    config_file = "src/model_utils/icnet.yaml"
    config_path = os.path.join(args_opt.project_path, config_file)
    with open(config_path, "r") as yaml_file:
        cfg = yaml.load(yaml_file.read())
    logger = SetupLogger(name="semantic_segmentation",
                         save_dir=cfg["train"]["ckpt_dir"],
                         distributed_rank=0,
                         filename='{}_{}_evaluate_log.txt'.format(cfg["model"]["name"], cfg["model"]["backbone"]))

    evaluator = Evaluator(cfg)
    evaluator.eval()
