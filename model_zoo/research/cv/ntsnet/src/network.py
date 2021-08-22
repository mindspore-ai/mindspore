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
"""ntsnet network wrapper."""
import math
import os
import time
import threading
import numpy as np
from mindspore import ops, load_checkpoint, load_param_into_net, Tensor, nn
from mindspore.ops import functional as F
from mindspore.ops import operations as P
import mindspore.context as context
import mindspore.common.dtype as mstype
from mindspore.train.callback import Callback
from mindspore.train.callback._callback import set_cur_net
from mindspore.train.callback._checkpoint import _check_file_name_prefix, _cur_dir, CheckpointConfig, CheckpointManager, \
    _chg_ckpt_file_name_if_same_exist
from mindspore.train._utils import _make_directory
from mindspore.train.serialization import save_checkpoint, _save_graph
from mindspore.parallel._ps_context import _is_role_pserver, _get_ps_mode_rank
from src.resnet import resnet50
from src.config import config

m_for_scrutinizer = config.m_for_scrutinizer
K = config.topK
input_size = config.input_size
num_classes = config.num_classes
lossLogName = config.lossLogName


def _fc(in_channel, out_channel):
    '''Weight init for dense cell'''
    stdv = 1 / math.sqrt(in_channel)
    weight = Tensor(np.random.uniform(-stdv, stdv, (out_channel, in_channel)).astype(np.float32))
    bias = Tensor(np.random.uniform(-stdv, stdv, (out_channel)).astype(np.float32))
    return nn.Dense(in_channel, out_channel, has_bias=True,
                    weight_init=weight, bias_init=bias).to_float(mstype.float32)


def _conv(in_channels, out_channels, kernel_size=3, stride=1, padding=0, pad_mode='pad'):
    """Conv2D wrapper."""
    shape = (out_channels, in_channels, kernel_size, kernel_size)
    stdv = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
    weights = Tensor(np.random.uniform(-stdv, stdv, shape).astype(np.float32))
    shape_bias = (out_channels,)
    biass = Tensor(np.random.uniform(-stdv, stdv, shape_bias).astype(np.float32))
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     pad_mode=pad_mode, weight_init=weights, has_bias=True, bias_init=biass)


_default_anchors_setting = (
    dict(layer='p3', stride=32, size=48, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
    dict(layer='p4', stride=64, size=96, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
    dict(layer='p5', stride=128, size=192, scale=[1, 2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
)


def generate_default_anchor_maps(anchors_setting=None, input_shape=input_size):
    """
    generate default anchor

    :param anchors_setting: all information of anchors
    :param input_shape: shape of input images, e.g. (h, w)
    :return: center_anchors: # anchors * 4 (oy, ox, h, w)
             edge_anchors: # anchors * 4 (y0, x0, y1, x1)
             anchor_area: # anchors * 1 (area)
    """
    if anchors_setting is None:
        anchors_setting = _default_anchors_setting

    center_anchors = np.zeros((0, 4), dtype=np.float32)
    edge_anchors = np.zeros((0, 4), dtype=np.float32)
    anchor_areas = np.zeros((0,), dtype=np.float32)
    input_shape = np.array(input_shape, dtype=int)

    for anchor_info in anchors_setting:
        stride = anchor_info['stride']
        size = anchor_info['size']
        scales = anchor_info['scale']
        aspect_ratios = anchor_info['aspect_ratio']

        output_map_shape = np.ceil(input_shape.astype(np.float32) / stride)
        output_map_shape = output_map_shape.astype(np.int)
        output_shape = tuple(output_map_shape) + (4,)
        ostart = stride / 2.
        oy = np.arange(ostart, ostart + stride * output_shape[0], stride)
        oy = oy.reshape(output_shape[0], 1)
        ox = np.arange(ostart, ostart + stride * output_shape[1], stride)
        ox = ox.reshape(1, output_shape[1])
        center_anchor_map_template = np.zeros(output_shape, dtype=np.float32)
        center_anchor_map_template[:, :, 0] = oy
        center_anchor_map_template[:, :, 1] = ox
        for scale in scales:
            for aspect_ratio in aspect_ratios:
                center_anchor_map = center_anchor_map_template.copy()
                center_anchor_map[:, :, 2] = size * scale / float(aspect_ratio) ** 0.5
                center_anchor_map[:, :, 3] = size * scale * float(aspect_ratio) ** 0.5
                edge_anchor_map = np.concatenate((center_anchor_map[..., :2] - center_anchor_map[..., 2:4] / 2.,
                                                  center_anchor_map[..., :2] + center_anchor_map[..., 2:4] / 2.),
                                                 axis=-1)
                anchor_area_map = center_anchor_map[..., 2] * center_anchor_map[..., 3]
                center_anchors = np.concatenate((center_anchors, center_anchor_map.reshape(-1, 4)))
                edge_anchors = np.concatenate((edge_anchors, edge_anchor_map.reshape(-1, 4)))
                anchor_areas = np.concatenate((anchor_areas, anchor_area_map.reshape(-1)))
    return center_anchors, edge_anchors, anchor_areas


class Navigator(nn.Cell):
    """Navigator"""

    def __init__(self):
        """Navigator init"""
        super(Navigator, self).__init__()
        self.down1 = _conv(2048, 128, 3, 1, padding=1, pad_mode='pad')
        self.down2 = _conv(128, 128, 3, 2, padding=1, pad_mode='pad')
        self.down3 = _conv(128, 128, 3, 2, padding=1, pad_mode='pad')
        self.ReLU = nn.ReLU()
        self.tidy1 = _conv(128, 6, 1, 1, padding=0, pad_mode='same')
        self.tidy2 = _conv(128, 6, 1, 1, padding=0, pad_mode='same')
        self.tidy3 = _conv(128, 9, 1, 1, padding=0, pad_mode='same')
        self.opConcat = ops.Concat(axis=1)
        self.opReshape = ops.Reshape()

    def construct(self, x):
        """Navigator construct"""
        batch_size = x.shape[0]
        d1 = self.ReLU(self.down1(x))
        d2 = self.ReLU(self.down2(d1))
        d3 = self.ReLU(self.down3(d2))
        t1 = self.tidy1(d1)
        t2 = self.tidy2(d2)
        t3 = self.tidy3(d3)
        t1 = self.opReshape(t1, (batch_size, -1, 1))
        t2 = self.opReshape(t2, (batch_size, -1, 1))
        t3 = self.opReshape(t3, (batch_size, -1, 1))
        return self.opConcat((t1, t2, t3))


class NTS_NET(nn.Cell):
    """Ntsnet"""

    def __init__(self, topK=6, resnet50Path=""):
        """Ntsnet init"""
        super(NTS_NET, self).__init__()
        feature_extractor = resnet50(1001)
        if resnet50Path != "":
            param_dict = load_checkpoint(resnet50Path)
            load_param_into_net(feature_extractor, param_dict)
        self.feature_extractor = feature_extractor  # Backbone
        self.feature_extractor.end_point = _fc(512 * 4, num_classes)
        self.navigator = Navigator()  # Navigator
        self.topK = topK
        self.num_classes = num_classes
        self.scrutinizer = _fc(2048 * (m_for_scrutinizer + 1), num_classes)  # Scrutinizer
        self.teacher = _fc(512 * 4, num_classes)  # Teacher
        _, edge_anchors, _ = generate_default_anchor_maps()
        self.pad_side = 224
        self.Pad_ops = ops.Pad(((0, 0), (0, 0), (self.pad_side, self.pad_side), (self.pad_side, self.pad_side)))
        self.np_edge_anchors = edge_anchors + 224
        self.edge_anchors = Tensor(self.np_edge_anchors, mstype.float32)
        self.opzeros = ops.Zeros()
        self.opones = ops.Ones()
        self.concat_op = ops.Concat(axis=1)
        self.nms = P.NMSWithMask(0.25)
        self.topK_op = ops.TopK(sorted=True)
        self.opReshape = ops.Reshape()
        self.opResizeLinear = ops.ResizeBilinear((224, 224))
        self.transpose = ops.Transpose()
        self.opsCropResize = ops.CropAndResize(method="bilinear_v2")
        self.min_float_num = -65536.0
        self.selected_mask_shape = (1614,)
        self.unchosen_score = Tensor(self.min_float_num * np.ones(self.selected_mask_shape, np.float32),
                                     mstype.float32)
        self.gatherND = ops.GatherNd()
        self.gatherD = ops.GatherD()
        self.squeezeop = P.Squeeze()
        self.select = P.Select()
        self.perm = (1, 2, 0)
        self.box_index = self.opzeros(((K,)), mstype.int32)
        self.crop_size = (224, 224)
        self.perm2 = (0, 3, 1, 2)
        self.m_for_scrutinizer = m_for_scrutinizer
        self.sortop = ops.Sort(descending=True)
        self.stackop = ops.Stack()

    def construct(self, x):
        """Ntsnet construct"""
        resnet_out, rpn_feature, feature = self.feature_extractor(x)
        x_pad = self.Pad_ops(x)
        batch_size = x.shape[0]
        rpn_feature = F.stop_gradient(rpn_feature)
        rpn_score = self.navigator(rpn_feature)
        edge_anchors = self.edge_anchors
        top_k_info = []
        current_img_for_teachers = []
        for i in range(batch_size):
            # using navigator output as scores to nms anchors
            rpn_score_current_img = self.opReshape(rpn_score[i:i + 1:1, ::], (-1, 1))
            bbox_score = self.squeezeop(rpn_score_current_img)
            bbox_score_sorted, bbox_score_sorted_indices = self.sortop(bbox_score)
            bbox_score_sorted_concat = self.opReshape(bbox_score_sorted, (-1, 1))
            edge_anchors_sorted_concat = self.gatherND(edge_anchors,
                                                       self.opReshape(bbox_score_sorted_indices, (1614, 1)))
            bbox = self.concat_op((edge_anchors_sorted_concat, bbox_score_sorted_concat))
            _, _, selected_mask = self.nms(bbox)
            selected_mask = F.stop_gradient(selected_mask)
            bbox_score = self.squeezeop(bbox_score_sorted_concat)
            scores_using = self.select(selected_mask, bbox_score, self.unchosen_score)
            # select the topk anchors and scores after nms
            _, topK_indices = self.topK_op(scores_using, self.topK)
            topK_indices = self.opReshape(topK_indices, (K, 1))
            bbox_topk = self.gatherND(bbox, topK_indices)
            top_k_info.append(self.opReshape(bbox_topk[::, 4:5:1], (-1,)))
            # crop from x_pad and resize to a fixed size using bilinear
            temp_pad = self.opReshape(x_pad[i:i + 1:1, ::, ::, ::], (3, 896, 896))
            temp_pad = self.transpose(temp_pad, self.perm)
            tensor_image = self.opReshape(temp_pad, (1,) + temp_pad.shape)
            tensor_box = self.gatherND(edge_anchors_sorted_concat, topK_indices)
            tensor_box = tensor_box / 895
            current_img_for_teacher = self.opsCropResize(tensor_image, tensor_box, self.box_index, self.crop_size)
            # the image cropped will be used to extractor feature and calculate loss
            current_img_for_teacher = self.opReshape(current_img_for_teacher, (-1, 224, 224, 3))
            current_img_for_teacher = self.transpose(current_img_for_teacher, self.perm2)
            current_img_for_teacher = self.opReshape(current_img_for_teacher, (-1, 3, 224, 224))
            current_img_for_teachers.append(current_img_for_teacher)
        feature = self.opReshape(feature, (batch_size, 1, -1))
        top_k_info = self.stackop(top_k_info)
        top_k_info = self.opReshape(top_k_info, (batch_size, self.topK))
        current_img_for_teachers = self.stackop(current_img_for_teachers)
        current_img_for_teachers = self.opReshape(current_img_for_teachers, (batch_size * self.topK, 3, 224, 224))
        current_img_for_teachers = F.stop_gradient(current_img_for_teachers)
        # extracor features of topk cropped images
        _, _, pre_teacher_features = self.feature_extractor(current_img_for_teachers)
        pre_teacher_features = self.opReshape(pre_teacher_features, (batch_size, self.topK, 2048))
        pre_scrutinizer_features = pre_teacher_features[::, 0:self.m_for_scrutinizer:1, ::]
        pre_scrutinizer_features = self.opReshape(pre_scrutinizer_features, (batch_size, self.m_for_scrutinizer, 2048))
        pre_scrutinizer_features = self.opReshape(self.concat_op((pre_scrutinizer_features, feature)), (batch_size, -1))
        # using topk cropped images, feed in scrutinzer and teacher, calculate loss
        scrutinizer_out = self.scrutinizer(pre_scrutinizer_features)
        teacher_out = self.teacher(pre_teacher_features)
        return resnet_out, scrutinizer_out, teacher_out, top_k_info
        # (batch_size, 200),(batch_size, 200),(batch_size,6, 200),(batch_size,6)


class WithLossCell(nn.Cell):
    """WithLossCell wrapper for ntsnet"""

    def __init__(self, backbone, loss_fn):
        """WithLossCell init"""
        super(WithLossCell, self).__init__(auto_prefix=True)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self.oneTensor = Tensor(1.0, mstype.float32)
        self.zeroTensor = Tensor(0.0, mstype.float32)
        self.opReshape = ops.Reshape()
        self.opOnehot = ops.OneHot()
        self.oplogsoftmax = ops.LogSoftmax()
        self.opZeros = ops.Zeros()
        self.opOnes = ops.Ones()
        self.opRelu = ops.ReLU()
        self.opGatherD = ops.GatherD()
        self.squeezeop = P.Squeeze()
        self.reducesumop = ops.ReduceSum()
        self.oprepeat = ops.repeat_elements
        self.cast = ops.Cast()

    def construct(self, image_data, label):
        """WithLossCell construct"""
        batch_size = image_data.shape[0]
        origin_label = label
        labelx = self.opReshape(label, (-1, 1))
        origin_label_repeatk_2D = self.oprepeat(labelx, rep=K, axis=1)
        origin_label_repeatk = self.opReshape(origin_label_repeatk_2D, (-1,))
        origin_label_repeatk_unsqueeze = self.opReshape(origin_label_repeatk_2D, (-1, 1))
        resnet_out, scrutinizer_out, teacher_out, top_k_info = self._backbone(image_data)
        teacher_out = self.opReshape(teacher_out, (batch_size * K, -1))
        log_softmax_teacher_out = -1 * self.oplogsoftmax(teacher_out)
        log_softmax_teacher_out_result = self.opGatherD(log_softmax_teacher_out, 1, origin_label_repeatk_unsqueeze)
        log_softmax_teacher_out_result = self.opReshape(log_softmax_teacher_out_result, (batch_size, K))
        oneHotLabel = self.opOnehot(origin_label, num_classes, self.oneTensor, self.zeroTensor)
        # using resnet_out to calculate resnet_real_out_loss
        resnet_real_out_loss = self._loss_fn(resnet_out, oneHotLabel)
        # using scrutinizer_out to calculate scrutinizer_out_loss
        scrutinizer_out_loss = self._loss_fn(scrutinizer_out, oneHotLabel)
        # using teacher_out and top_k_info to calculate ranking loss
        loss = self.opZeros((), mstype.float32)
        num = top_k_info.shape[0]
        for i in range(K):
            log_softmax_teacher_out_inlabel_unsqueeze = self.opReshape(log_softmax_teacher_out_result[::, i:i + 1:1],
                                                                       (-1, 1))
            compareX = log_softmax_teacher_out_result > log_softmax_teacher_out_inlabel_unsqueeze
            pivot = self.opReshape(top_k_info[::, i:i + 1:1], (-1, 1))
            information = 1 - pivot + top_k_info
            loss_p = information * compareX
            loss_p_temp = self.opRelu(loss_p)
            loss_p = self.reducesumop(loss_p_temp)
            loss += loss_p
        rank_loss = loss / num
        oneHotLabel2 = self.opOnehot(origin_label_repeatk, num_classes, self.oneTensor, self.zeroTensor)
        # using teacher_out to calculate teacher_loss
        teacher_loss = self._loss_fn(teacher_out, oneHotLabel2)
        total_loss = resnet_real_out_loss + rank_loss + scrutinizer_out_loss + teacher_loss
        return total_loss

    @property
    def backbone_network(self):
        """WithLossCell backbone"""
        return self._backbone


class ModelCheckpoint(Callback):
    """
    The checkpoint callback class.
    It is called to combine with train process and save the model and network parameters after training.
    Note:
        In the distributed training scenario, please specify different directories for each training process
        to save the checkpoint file. Otherwise, the training may fail.
    Args:
        prefix (str): The prefix name of checkpoint files. Default: "CKP".
        directory (str): The path of the folder which will be saved in the checkpoint file. Default: None.
        ckconfig (CheckpointConfig): Checkpoint strategy configuration. Default: None.
    Raises:
        ValueError: If the prefix is invalid.
        TypeError: If the config is not CheckpointConfig type.
    """

    def __init__(self, prefix='CKP', directory=None, ckconfig=None,
                 device_num=1, device_id=0, args=None, run_modelart=False):
        super(ModelCheckpoint, self).__init__()
        self._latest_ckpt_file_name = ""
        self._init_time = time.time()
        self._last_time = time.time()
        self._last_time_for_keep = time.time()
        self._last_triggered_step = 0
        self.run_modelart = run_modelart
        if _check_file_name_prefix(prefix):
            self._prefix = prefix
        else:
            raise ValueError("Prefix {} for checkpoint file name invalid, "
                             "please check and correct it and then continue.".format(prefix))
        if directory is not None:
            self._directory = _make_directory(directory)
        else:
            self._directory = _cur_dir
        if ckconfig is None:
            self._config = CheckpointConfig()
        else:
            if not isinstance(ckconfig, CheckpointConfig):
                raise TypeError("ckconfig should be CheckpointConfig type.")
            self._config = ckconfig
        # get existing checkpoint files
        self._manager = CheckpointManager()
        self._prefix = _chg_ckpt_file_name_if_same_exist(self._directory, self._prefix)
        self._graph_saved = False
        self._need_flush_from_cache = True
        self.device_num = device_num
        self.device_id = device_id
        self.args = args

    def step_end(self, run_context):
        """
        Save the checkpoint at the end of step.
        Args:
            run_context (RunContext): Context of the train running.
        """
        if _is_role_pserver():
            self._prefix = "PServer_" + str(_get_ps_mode_rank()) + "_" + self._prefix
        cb_params = run_context.original_args()
        _make_directory(self._directory)
        # save graph (only once)
        if not self._graph_saved:
            graph_file_name = os.path.join(self._directory, self._prefix + '-graph.meta')
            if os.path.isfile(graph_file_name) and context.get_context("mode") == context.GRAPH_MODE:
                os.remove(graph_file_name)
            _save_graph(cb_params.train_network, graph_file_name)
            self._graph_saved = True
        thread_list = threading.enumerate()
        for thread in thread_list:
            if thread.getName() == "asyn_save_ckpt":
                thread.join()
        self._save_ckpt(cb_params)

    def end(self, run_context):
        """
        Save the last checkpoint after training finished.
        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        _to_save_last_ckpt = True
        self._save_ckpt(cb_params, _to_save_last_ckpt)
        thread_list = threading.enumerate()
        for thread in thread_list:
            if thread.getName() == "asyn_save_ckpt":
                thread.join()
        from mindspore.parallel._cell_wrapper import destroy_allgather_cell
        destroy_allgather_cell()

    def _check_save_ckpt(self, cb_params, force_to_save):
        """Check whether save checkpoint files or not."""
        if self._config.save_checkpoint_steps and self._config.save_checkpoint_steps > 0:
            if cb_params.cur_step_num >= self._last_triggered_step + self._config.save_checkpoint_steps \
                    or force_to_save is True:
                return True
        elif self._config.save_checkpoint_seconds and self._config.save_checkpoint_seconds > 0:
            self._cur_time = time.time()
            if (self._cur_time - self._last_time) > self._config.save_checkpoint_seconds or force_to_save is True:
                self._last_time = self._cur_time
                return True
        return False

    def _save_ckpt(self, cb_params, force_to_save=False):
        """Save checkpoint files."""
        if cb_params.cur_step_num == self._last_triggered_step:
            return
        save_ckpt = self._check_save_ckpt(cb_params, force_to_save)
        step_num_in_epoch = int((cb_params.cur_step_num - 1) % cb_params.batch_num + 1)
        if save_ckpt:
            cur_ckpoint_file = self._prefix + "-" + str(cb_params.cur_epoch_num) + "_" \
                               + str(step_num_in_epoch) + ".ckpt"
            # update checkpoint file list.
            self._manager.update_ckpoint_filelist(self._directory, self._prefix)
            # keep checkpoint files number equal max number.
            if self._config.keep_checkpoint_max and \
                    0 < self._config.keep_checkpoint_max <= self._manager.ckpoint_num:
                self._manager.remove_oldest_ckpoint_file()
            elif self._config.keep_checkpoint_per_n_minutes and \
                    self._config.keep_checkpoint_per_n_minutes > 0:
                self._cur_time_for_keep = time.time()
                if (self._cur_time_for_keep - self._last_time_for_keep) \
                        < self._config.keep_checkpoint_per_n_minutes * 60:
                    self._manager.keep_one_ckpoint_per_minutes(self._config.keep_checkpoint_per_n_minutes,
                                                               self._cur_time_for_keep)
            # generate the new checkpoint file and rename it.
            cur_file = os.path.join(self._directory, cur_ckpoint_file)
            self._last_time_for_keep = time.time()
            self._last_triggered_step = cb_params.cur_step_num
            if context.get_context("enable_ge"):
                set_cur_net(cb_params.train_network)
                cb_params.train_network.exec_checkpoint_graph()
            network = self._config.saved_network if self._config.saved_network is not None \
                else cb_params.train_network
            save_checkpoint(network, cur_file, self._config.integrated_save,
                            self._config.async_save)
            self._latest_ckpt_file_name = cur_file
            if self.run_modelart and (self.device_num == 1 or self.device_id == 0):
                import moxing as mox
                mox.file.copy_parallel(src_url=cur_file, dst_url=os.path.join(self.args.train_url, cur_ckpoint_file))

    def _flush_from_cache(self, cb_params):
        """Flush cache data to host if tensor is cache enable."""
        has_cache_params = False
        params = cb_params.train_network.get_parameters()
        for param in params:
            if param.cache_enable:
                has_cache_params = True
                Tensor(param).flush_from_cache()
        if not has_cache_params:
            self._need_flush_from_cache = False

    @property
    def latest_ckpt_file_name(self):
        """Return the latest checkpoint path and file name."""
        return self._latest_ckpt_file_name


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss is NAN or INF terminating training.
    Note:
        If per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """

    def __init__(self, per_print_times=1, rank_id=0, local_output_url="",
                 device_num=1, device_id=0, args=None, run_modelart=False):
        super(LossCallBack, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("print_step must be int and >= 0.")
        self._per_print_times = per_print_times
        self.count = 0
        self.rpn_loss_sum = 0
        self.rpn_cls_loss_sum = 0
        self.rpn_reg_loss_sum = 0
        self.rank_id = rank_id
        self.local_output_url = local_output_url
        self.device_num = device_num
        self.device_id = device_id
        self.args = args
        self.time_stamp_first = time.time()
        self.run_modelart = run_modelart

    def step_end(self, run_context):
        """
            Called after each step finished.
            Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        rpn_loss = cb_params.net_outputs.asnumpy()
        self.count += 1
        self.rpn_loss_sum += float(rpn_loss)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
        if self.count >= 1:
            time_stamp_current = time.time()
            rpn_loss = self.rpn_loss_sum / self.count
            loss_file = open(os.path.join(self.local_output_url, lossLogName), "a+")
            loss_file.write("%lu epoch: %s step: %s ,rpn_loss: %.5f" %
                            (time_stamp_current - self.time_stamp_first, cb_params.cur_epoch_num, cur_step_in_epoch,
                             rpn_loss))
            loss_file.write("\n")
            loss_file.close()
            if self.run_modelart and (self.device_num == 1 or self.device_id == 0):
                import moxing as mox
                mox.file.copy_parallel(src_url=os.path.join(self.local_output_url, lossLogName),
                                       dst_url=os.path.join(self.args.train_url, lossLogName))
