# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
'''eval.py'''
import os
import argparse
from PIL import Image
from tabulate import tabulate

from mindspore import context
import mindspore.ops as ops
from mindspore.context import ParallelMode
from mindspore import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.dataset.transforms.py_transforms import Compose
from mindspore.dataset.vision.py_transforms import ToTensor, Normalize

from src.dataloader import create_CitySegmentation
from src.fast_scnn import FastSCNN
from src.score import SegmentationMetric
from src.logger import get_logger
import src.visualize as visualize

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Fast-SCNN on mindspore')
    parser.add_argument('--dataset', type=str, default='/data/dataset/citys/',
                        help='dataset name (default: /data/dataset/citys/)')
    parser.add_argument('--base_size', type=int, default=1024, help='base image size')
    parser.add_argument('--crop_size', type=int, default=(768, 768), help='crop image size')
    parser.add_argument('--train_split', type=str, default='train',
                        help='dataset train split (default: train)')
    parser.add_argument('--aux', action='store_true', default=True, help='Auxiliary loss')
    parser.add_argument('--aux_weight', type=float, default=0.4,
                        help='auxiliary loss weight')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--save_every', type=int, default=1, metavar='N',
                        help='save ckpt every N epoch')
    parser.add_argument('--resume_path', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--resume_name', type=str, default=None,
                        help='resuming file name')
    parser.add_argument('--batch_size', type=int, default=2, metavar='N',
                        help='input batch size for training (default: 2)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='base learning rate (default: 0.045)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=4e-5, metavar='M',
                        help='w-decay (default: 4e-5)')

    parser.add_argument('--eval_while_train', type=int, default=1, help='eval while training')
    parser.add_argument('--eval_steps', type=int, default=10, help='each N epochs we eval')
    parser.add_argument('--eval_start_epoch', type=int, default=850, help='eval_start_epoch')
    parser.add_argument('--use_modelarts', type=int, default=0,
                        help='when set True, we should load dataset from obs with moxing')
    parser.add_argument('--train_url', type=str, default='train_url/',
                        help='needed by modelarts, but we donot use it because the name is ambiguous')
    parser.add_argument('--data_url', type=str, default='data_url/',
                        help='needed by modelarts, but we donot use it because the name is ambiguous')
    parser.add_argument('--output_path', type=str, default='./outputs/',
                        help='output_path,when use_modelarts is set True, it will be cache/output/')
    parser.add_argument('--outer_path', type=str, default='s3://output/',
                        help='obs path,to store e.g ckpt files ')

    parser.add_argument('--device_target', type=str, default='Ascend',
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument('--is_distributed', type=int, default=0, help='if multi device')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')
    parser.add_argument('--is_save_on_master', type=int, default=1,
                        help='save ckpt on master or all rank')
    parser.add_argument('--ckpt_save_max', type=int, default=800,
                        help='Maximum number of checkpoint files can be saved. Default: 5.')
    # the parser
    args_ = parser.parse_args()
    return args_

args = parse_args()
save_dir = args.output_path
device_id = int(os.getenv('DEVICE_ID', '0'))
context.set_context(mode=context.GRAPH_MODE,
                    device_target=args.device_target, save_graphs=False)

def validation():
    '''validation'''
    if args.is_distributed:
        assert args.device_target == "Ascend"
        init()
        context.set_context(device_id=device_id)
        args.rank = get_rank()
        args.group_size = get_group_size()
        device_num = args.group_size
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          mirror_mean=True)
    else:
        if args.device_target in ["Ascend", "GPU"]:
            context.set_context(device_id=device_id)

    # select for master rank save ckpt or all rank save, compatible for model parallel
    args.rank_save_ckpt_flag = 0
    if args.is_save_on_master:
        if args.rank == 0:
            args.rank_save_ckpt_flag = 1
    else:
        args.rank_save_ckpt_flag = 1

    metric = SegmentationMetric(19)
    metric.reset()

    # create network
    model = FastSCNN(num_classes=19, aux=True)
    if args.resume_path:
        if args.use_modelarts:
            import moxing as mox
            args.logger.info("copying resume checkpoint from obs to cache....")
            mox.file.copy_parallel(args.resume_path, 'cache/resume_path')
            args.logger.info("copying resume checkpoint finished....")
            args.resume_path = 'cache/resume_path/'

        args.resume_path = os.path.join(args.resume_path, args.resume_name)
        args.logger.info('loading resume checkpoint {} into network'.format(args.resume_path))
        load_param_into_net(model, load_checkpoint(args.resume_path))
        args.logger.info('loaded resume checkpoint {} into network'.format(args.resume_path))

    model.set_train(False)
    # image transform
    input_transform = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    if args.use_modelarts:
        import moxing as mox
        args.logger.info("copying dataset from obs to cache....")
        mox.file.copy_parallel(args.dataset, 'cache/dataset')
        args.logger.info("copying dataset finished....")
        args.dataset = 'cache/dataset/'

    val_dataset, _ = create_CitySegmentation(args, data_path=args.dataset, \
                                       split='val', mode='val', transform=input_transform, \
                                       base_size=args.base_size, crop_size=args.crop_size, \
                                       batch_size=1, device_num=args.group_size, \
                                       rank=args.rank, shuffle=False)
    classes = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
               'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
               'truck', 'bus', 'train', 'motorcycle', 'bicycle')

    data_loader = val_dataset.create_dict_iterator()
    for i, data in enumerate(data_loader):
        images = data["image"]
        targets = data["label"]
        output = model(images)[0]
        metric.update(output, targets)
        pixAcc, mIoU = metric.get()
        args.logger.info("[EVAL] Sample: {:d}, pixAcc: {:.3f}, mIoU: {:.3f}".format(i + 1, pixAcc * 100, mIoU * 100))

        output = ops.Argmax(axis=0)(output[0]).asnumpy()
        out_img = Image.fromarray(output.astype('uint8'))
        out_img.putpalette(visualize.cityspallete)
        outname = str(i) + '.png'
        out_img.save(os.path.join(save_dir, outname))

    pixAcc, mIoU = metric.get()
    args.logger.info("[EVAL END] pixAcc: {:.3f}, mIoU: {:.3f}".format(pixAcc * 100, mIoU * 100))

    pixAcc, mIoU, category_iou = metric.get(return_category_iou=True)
    args.logger.info('End validation pixAcc: {:.3f}, mIoU: {:.3f}'.format(pixAcc * 100, mIoU * 100))
    txtName = os.path.join(save_dir, "eval_results.txt")
    with open(txtName, "w") as f:
        string = 'validation pixAcc:' + str(pixAcc * 100) + ', mIoU:' + str(mIoU * 100)
        f.write(string)
        f.write('\n')
        headers = ['class id', 'class name', 'iou']
        table = []
        for i, cls_name in enumerate(classes):
            table.append([cls_name, category_iou[i]])
            string = 'class name: ' + cls_name + ' iou: ' + str(category_iou[i]) + '\n'
            f.write(string)
        args.logger.info('Category iou: \n {}'.format(tabulate(table, headers, \
                               tablefmt='grid', showindex="always", numalign='center', stralign='center')))

if __name__ == '__main__':
    args.logger = get_logger(save_dir, "Fast_SCNN", args.rank)
    args.logger.save_args(args)
    validation()
