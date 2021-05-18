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
######################## eval DDM example ########################
eval DDM according to model file:
python eval.py --data_path /YourDataPath --pretrained Your.ckpt
"""

import os
import argparse
import numpy as np
import mindspore.dataset as ds
from mindspore.ops import ResizeBilinear
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore import context, Tensor
import mindspore.common.dtype as mstype
from dataset import init_vid_dataset
from config import cfg
from net import deeplabv2_mindspore
from utils.func import per_class_iu, fast_hist

parser = argparse.ArgumentParser(description='Check_Point File Path')
parser.add_argument('--pretrained', type=str)
parser.add_argument("--data_path", type=str)
args = parser.parse_known_args()[0]

net = deeplabv2_mindspore.get_deeplab_v2()
param_dict = load_checkpoint(args.pretrained)
load_param_into_net(net, param_dict)

test_dataset = init_vid_dataset(name=cfg.TEST.DATA,
                                root=args.data_path,
                                list_path=cfg.TEST.DATA_LIST,
                                num_classes=cfg.NUM_CLASSES,
                                set_name=cfg.TEST.SET,
                                info_path=cfg.TEST.INFO,
                                crop_size=cfg.TEST.INPUT_SIZE,
                                mean=cfg.TEST.IMG_MEAN,
                                labels_size=cfg.TEST.OUTPUT_SIZE)

test_loader = ds.GeneratorDataset(test_dataset,
                                  num_parallel_workers=1,
                                  shuffle=False,
                                  column_names=["data", "label"])

test_loader = test_loader.batch(cfg.TEST.BATCH_SIZE, drop_remainder=True)

def evaluate(model, testloader, num_class, fixed_test_size=True, verbose=True):
    """
    Evaluation during training.
    """
    hist = np.zeros((cfg.NUM_CLASSES, cfg.NUM_CLASSES))
    test_iter = testloader.create_dict_iterator(output_numpy=True)

    nt = 0
    for ti in test_iter:
        image, label = Tensor(ti["data"], mstype.float32), Tensor(ti["label"], mstype.float32)
        if not fixed_test_size:
            interp = ResizeBilinear(size=(label.shape[1], label.shape[2]), align_corners=True)
        else:
            interp = ResizeBilinear(size=(cfg.TEST.OUTPUT_SIZE[1], cfg.TEST.OUTPUT_SIZE[0]), align_corners=True)
        pred_main = model(Tensor(image, mstype.float32))[1]
        output = interp(pred_main).asnumpy()[0]
        output = output.transpose((1, 2, 0))
        output = np.argmax(output, axis=2)
        label = label.asnumpy()[0]
        hist += fast_hist(label.flatten(), output.flatten(), cfg.NUM_CLASSES)
        if verbose and nt > 0 and nt % 100 == 0:
            print('{:d} : {:0.2f}'.format(
                nt, 100 * np.nanmean(per_class_iu(hist))))
        nt += 1
    inters_over_union_classes = per_class_iu(hist)
    # pickle_dump(all_res, cache_path)
    if cfg.NUM_CLASSES == 19:
        computed_miou_19 = round(np.nanmean(inters_over_union_classes) * 100, 2)
        computed_miou_16 = round(np.mean(inters_over_union_classes[[0, 1, 2, 3, 4, 5,\
                                 6, 7, 8, 10, 11, 12, 13, 15, 17, 18]]) * 100, 2)
        computed_miou_13 = round(np.mean(inters_over_union_classes[[0, 1, 2, 6, 7, 8,\
                                 10, 11, 12, 13, 15, 17, 18]]) * 100, 2)
    elif cfg.NUM_CLASSES == 16:
        computed_miou_19 = 0
        computed_miou_16 = round(np.nanmean(inters_over_union_classes) * 100, 2)
        computed_miou_13 = round(np.mean(inters_over_union_classes[[0, 1, 2, 6, 7, 8,\
                                 9, 10, 11, 12, 13, 14, 15]]) * 100, 2)
    print('==>Current mIoUs: \n', 'Class 19: ', computed_miou_19, '\n', 'Class 16: ', computed_miou_16,
          '\n', 'Class 13: ', computed_miou_13)
    if verbose:
        display_stats(num_class, inters_over_union_classes)
    return [computed_miou_19, computed_miou_16, computed_miou_13], inters_over_union_classes

def display_stats(num_class, inters_over_union_classes):
    """print classes' performance"""
    for ind_class in range(num_class):
        print(str(ind_class) + '\t' + str(round(inters_over_union_classes[ind_class] * 100, 2)))

context.set_context(mode=context.GRAPH_MODE,
                    device_target="Ascend",
                    save_graphs=False,
                    device_id=int(os.getenv("DEVICE_ID")))

evaluate(net, test_loader, cfg.NUM_CLASSES, fixed_test_size=True, verbose=True)
