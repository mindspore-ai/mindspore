# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import time
import matplotlib.pyplot as plt

from mindspore import context
from mindspore import Tensor
from mindspore.context import ParallelMode
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import dtype as mstype
import mindspore.dataset as de

from src.data_preprocess import SingleScaleTrans
from src.FaceDetection.yolov3 import HwYolov3 as backbone_HwYolov3
from src.FaceDetection import voc_wrapper
from src.network_define import BuildTestNetwork, get_bounding_boxes, tensor_to_brambox, \
    parse_gt_from_anno, parse_rets, calc_recall_precision_ap

from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id, get_device_num, get_rank_id


plt.switch_backend('agg')

def load_pretrain(net, cfg):
    '''load pretrain model'''
    if os.path.isfile(cfg.pretrained):
        param_dict = load_checkpoint(cfg.pretrained)
        param_dict_new = {}
        for key, values in param_dict.items():
            if key.startswith('moments.'):
                continue
            elif key.startswith('network.'):
                param_dict_new[key[8:]] = values
            else:
                param_dict_new[key] = values
        load_param_into_net(net, param_dict_new)
        print('load model {} success'.format(cfg.pretrained))
    else:
        print('load model {} failed, please check the path of model, evaluating end'.format(cfg.pretrained))
        exit(0)

    return net

def modelarts_pre_process():
    '''modelarts pre process function.'''
    def unzip(zip_file, save_dir):
        import zipfile
        s_time = time.time()
        if not os.path.exists(os.path.join(save_dir, config.modelarts_dataset_unzip_name)):
            zip_isexist = zipfile.is_zipfile(zip_file)
            if zip_isexist:
                fz = zipfile.ZipFile(zip_file, 'r')
                data_num = len(fz.namelist())
                print("Extract Start...")
                print("unzip file num: {}".format(data_num))
                data_print = int(data_num / 100) if data_num > 100 else 1
                i = 0
                for file in fz.namelist():
                    if i % data_print == 0:
                        print("unzip percent: {}%".format(int(i * 100 / data_num)), flush=True)
                    i += 1
                    fz.extract(file, save_dir)
                print("cost time: {}min:{}s.".format(int((time.time() - s_time) / 60),
                                                     int(int(time.time() - s_time) % 60)))
                print("Extract Done.")
            else:
                print("This is not zip.")
        else:
            print("Zip has been extracted.")

    if config.need_modelarts_dataset_unzip:
        zip_file_1 = os.path.join(config.data_path, config.modelarts_dataset_unzip_name + ".zip")
        save_dir_1 = os.path.join(config.data_path)

        sync_lock = "/tmp/unzip_sync.lock"

        # Each server contains 8 devices as most.
        if get_device_id() % min(get_device_num(), 8) == 0 and not os.path.exists(sync_lock):
            print("Zip file path: ", zip_file_1)
            print("Unzip file save dir: ", save_dir_1)
            unzip(zip_file_1, save_dir_1)
            print("===Finish extract data synchronization===")
            try:
                os.mknod(sync_lock)
            except IOError:
                pass

        while True:
            if os.path.exists(sync_lock):
                break
            time.sleep(1)

        print("Device: {}, Finish sync unzip data from {} to {}.".format(get_device_id(), zip_file_1, save_dir_1))

    config.result_path = os.path.join(config.output_path, "results")


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_eval():
    '''run eval'''
    config.world_size = get_device_num()
    config.local_rank = get_rank_id()
    devid = get_device_id() if config.run_platform != 'CPU' else 0
    context.set_context(mode=context.GRAPH_MODE, device_target=config.run_platform, save_graphs=False, device_id=devid)
    print('=============yolov3 start evaluating==================')

    context.set_auto_parallel_context(parallel_mode=ParallelMode.STAND_ALONE, device_num=config.world_size,
                                      gradients_mean=True)

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
    print('Loading data from {}'.format(config.mindrecord_path))
    ds = de.MindDataset(config.mindrecord_path + "0", columns_list=["image", "annotation", "image_name", "image_size"])

    single_scale_trans = SingleScaleTrans(resize=config.input_shape)
    ds = ds.batch(config.batch_size, per_batch_map=single_scale_trans,
                  input_columns=["image", "annotation", "image_name", "image_size"], num_parallel_workers=8)

    config.steps_per_epoch = ds.get_dataset_size()

    # backbone
    network = backbone_HwYolov3(num_classes, num_anchors_list, config)
    network = load_pretrain(network, config)

    ds = ds.repeat(1)

    det = {}
    img_size = {}
    img_anno = {}

    model_name = config.pretrained.split('/')[-1].replace('.ckpt', '')
    result_path = os.path.join(config.result_path, model_name)
    if os.path.exists(result_path):
        pass
    if not os.path.isdir(result_path):
        os.makedirs(result_path, exist_ok=True)

    # result file
    ret_files_set = {'face': os.path.join(result_path, 'comp4_det_test_face_rm5050.txt'),}

    test_net = BuildTestNetwork(network, reduction_0, reduction_1, reduction_2, anchors, anchors_mask, num_classes,
                                config)

    print('conf_thresh:', config.conf_thresh)

    eval_times = 0

    for data in ds.create_tuple_iterator(output_numpy=True):
        batch_images, batch_labels, batch_image_name, batch_image_size = data[0:4]
        eval_times += 1

        img_tensor = Tensor(batch_images, mstype.float32)

        dets = []
        tdets = []

        coords_0, cls_scores_0, coords_1, cls_scores_1, coords_2, cls_scores_2 = test_net(img_tensor)

        boxes_0, boxes_1, boxes_2 = get_bounding_boxes(coords_0, cls_scores_0, coords_1, cls_scores_1, coords_2,
                                                       cls_scores_2, config.conf_thresh, config.input_shape,
                                                       num_classes)

        converted_boxes_0, converted_boxes_1, converted_boxes_2 = tensor_to_brambox(boxes_0, boxes_1, boxes_2,
                                                                                    config.input_shape, labels)

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
    print('batch size: ', config.batch_size)

    netw, neth = config.input_shape
    reorg_dets = voc_wrapper.reorg_detection(det, netw, neth, img_size)
    voc_wrapper.gen_results(reorg_dets, result_path, img_size, config.nms_thresh)

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

if __name__ == "__main__":
    run_eval()
