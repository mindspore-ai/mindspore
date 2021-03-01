# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Face detection eval."""
import os
import argparse
import matplotlib.pyplot as plt

from mindspore import context
from mindspore import Tensor
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import dtype as mstype
import mindspore.dataset as de




from src.data_preprocess import SingleScaleTrans
from src.config import config
from src.FaceDetection.yolov3 import HwYolov3 as backbone_HwYolov3
from src.FaceDetection import voc_wrapper
from src.network_define import BuildTestNetwork, get_bounding_boxes, tensor_to_brambox, \
    parse_gt_from_anno, parse_rets, calc_recall_precision_ap

plt.switch_backend('agg')
devid = int(os.getenv('DEVICE_ID'))
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", save_graphs=False, device_id=devid)


def parse_args():
    '''parse_args'''
    parser = argparse.ArgumentParser('Yolov3 Face Detection')

    parser.add_argument('--mindrecord_path', type=str, default='', help='dataset path, e.g. /home/data.mindrecord')
    parser.add_argument('--pretrained', type=str, default='', help='pretrained model to load')
    parser.add_argument('--local_rank', type=int, default=0, help='current rank to support distributed')
    parser.add_argument('--world_size', type=int, default=1, help='current process number to support distributed')

    arg, _ = parser.parse_known_args()

    return arg


if __name__ == "__main__":
    args = parse_args()

    print('=============yolov3 start evaluating==================')

    # logger
    args.batch_size = config.batch_size
    args.input_shape = config.input_shape
    args.result_path = config.result_path
    args.conf_thresh = config.conf_thresh
    args.nms_thresh = config.nms_thresh

    context.set_auto_parallel_context(parallel_mode=ParallelMode.STAND_ALONE, device_num=args.world_size,
                                      gradients_mean=True)
    mindrecord_path = args.mindrecord_path
    print('Loading data from {}'.format(mindrecord_path))

    num_classes = config.num_classes
    if num_classes > 1:
        raise NotImplementedError('num_classes > 1: Yolov3 postprocess not implemented!')

    anchors = config.anchors
    anchors_mask = config.anchors_mask
    num_anchors_list = [len(x) for x in anchors_mask]

    reduction_0 = 64.0
    reduction_1 = 32.0
    reduction_2 = 16.0
    labels = ['face']
    classes = {0: 'face'}

    # dataloader
    ds = de.MindDataset(mindrecord_path + "0", columns_list=["image", "annotation", "image_name", "image_size"])

    single_scale_trans = SingleScaleTrans(resize=args.input_shape)

    ds = ds.batch(args.batch_size, per_batch_map=single_scale_trans,
                  input_columns=["image", "annotation", "image_name", "image_size"], num_parallel_workers=8)

    args.steps_per_epoch = ds.get_dataset_size()

    # backbone
    network = backbone_HwYolov3(num_classes, num_anchors_list, args)

    # load pretrain model
    if os.path.isfile(args.pretrained):
        param_dict = load_checkpoint(args.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(network, param_dict_new)
        print('load model {} success'.format(args.pretrained))
    else:
        print('load model {} failed, please check the path of model, evaluating end'.format(args.pretrained))
        exit(0)

    ds = ds.repeat(1)

    det = {}
    img_size = {}
    img_anno = {}

    model_name = args.pretrained.split('/')[-1].replace('.ckpt', '')
    result_path = os.path.join(args.result_path, model_name)
    if os.path.exists(result_path):
        pass
    if not os.path.isdir(result_path):
        os.makedirs(result_path, exist_ok=True)

    # result file
    ret_files_set = {
        'face': os.path.join(result_path, 'comp4_det_test_face_rm5050.txt'),
    }

    test_net = BuildTestNetwork(network, reduction_0, reduction_1, reduction_2, anchors, anchors_mask, num_classes,
                                args)

    print('conf_thresh:', args.conf_thresh)

    eval_times = 0

    for data in ds.create_tuple_iterator(output_numpy=True):
        batch_images = data[0]
        batch_labels = data[1]
        batch_image_name = data[2]
        batch_image_size = data[3]
        eval_times += 1

        img_tensor = Tensor(batch_images, mstype.float32)

        dets = []
        tdets = []

        coords_0, cls_scores_0, coords_1, cls_scores_1, coords_2, cls_scores_2 = test_net(img_tensor)

        boxes_0, boxes_1, boxes_2 = get_bounding_boxes(coords_0, cls_scores_0, coords_1, cls_scores_1, coords_2,
                                                       cls_scores_2, args.conf_thresh, args.input_shape,
                                                       num_classes)

        converted_boxes_0, converted_boxes_1, converted_boxes_2 = tensor_to_brambox(boxes_0, boxes_1, boxes_2,
                                                                                    args.input_shape, labels)

        tdets.append(converted_boxes_0)
        tdets.append(converted_boxes_1)
        tdets.append(converted_boxes_2)

        batch = len(tdets[0])
        for b in range(batch):
            single_dets = []
            for op in range(3):
                single_dets.extend(tdets[op][b])
            dets.append(single_dets)

        det.update({batch_image_name[k].decode('UTF-8'): v for k, v in enumerate(dets)})
        img_size.update({batch_image_name[k].decode('UTF-8'): v for k, v in enumerate(batch_image_size)})
        img_anno.update({batch_image_name[k].decode('UTF-8'): v for k, v in enumerate(batch_labels)})

    print('eval times:', eval_times)
    print('batch size: ', args.batch_size)

    netw, neth = args.input_shape
    reorg_dets = voc_wrapper.reorg_detection(det, netw, neth, img_size)
    voc_wrapper.gen_results(reorg_dets, result_path, img_size, args.nms_thresh)

    # compute mAP
    ground_truth = parse_gt_from_anno(img_anno, classes)

    ret_list = parse_rets(ret_files_set)
    iou_thr = 0.5
    evaluate = calc_recall_precision_ap(ground_truth, ret_list, iou_thr)

    aps_str = ''
    for cls in evaluate:
        per_line, = plt.plot(evaluate[cls]['recall'], evaluate[cls]['precision'], 'b-')
        per_line.set_label('%s:AP=%.3f' % (cls, evaluate[cls]['ap']))
        aps_str += '_%s_AP_%.3f' % (cls, evaluate[cls]['ap'])
        plt.plot([i / 1000.0 for i in range(1, 1001)], [i / 1000.0 for i in range(1, 1001)], 'y--')
        plt.axis([0, 1.2, 0, 1.2])
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.grid()

        plt.legend()
        plt.title('PR')

    # save mAP
    ap_save_path = os.path.join(result_path, result_path.replace('/', '_') + aps_str + '.png')
    print('Saving {}'.format(ap_save_path))
    plt.savefig(ap_save_path)

    print('=============yolov3 evaluating finished==================')
