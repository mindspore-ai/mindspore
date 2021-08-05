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
"""set tracker"""
import time
import numpy as np
import cv2
from tqdm import tqdm
import mindspore.numpy as ms_np
from mindspore import load_checkpoint, load_param_into_net, ops, Tensor
from .alexnet import SiameseAlexNet
from .config import config
from .utils import get_exemplar_image, get_pyramid_instance_image, show_image

class SiamFCTracker:
    """
        tracker used in evel
        Args:
            model_path : checkpoint path
    """
    def __init__(self, model_path, is_deterministic=True):
        self.network = SiameseAlexNet(train=False)
        load_param_into_net(self.network, load_checkpoint(model_path), strict_load=True)
        self.network.set_train(False)
        self.name = 'SiamFC'
        self.is_deterministic = is_deterministic

    def _cosine_window(self, size):
        """
            get the cosine window
        """
        cos_window = np.hanning(int(size[0]))[:, np.newaxis].dot(np.hanning(int(size[1]))
                                                                 [np.newaxis, :])
        cos_window = cos_window.astype(np.float32)
        cos_window /= np.sum(cos_window)
        return cos_window


    def init(self, frame, bbox):
        """ initialize siamfc trackers
        Args:
            frame: an RGB image
            bbox: one-based bounding box [x, y, width, height]
        """
        self.bbox = (bbox[0]-1, bbox[1]-1, bbox[0]-1+bbox[2], bbox[1]-1 + bbox[3])  # zero based
        self.pos = np.array(
            [bbox[0]-1+(bbox[2]-1)/2, bbox[1]-1+(bbox[3]-1)/2])  # center x, center y, zero based
        self.target_sz = np.array([bbox[2], bbox[3]])  # width, height
        # get exemplar img
        self.img_mean = tuple(map(int, frame.mean(axis=(0, 1))))
        exemplar_img, scale_z, s_z = get_exemplar_image(frame, self.bbox,
                                                        config.exemplar_size, config.context_amount,
                                                        self.img_mean)  # context_amount = 0.5
        # exemplar_size=127
        exemplar_img = np.transpose(exemplar_img, (2, 0, 1))[None, :, :, :]
        exemplar_img = Tensor(exemplar_img, dtype=ms_np.float32)
        none = ms_np.ones(1)
        self.exemplar = self.network(exemplar_img, none)
        self.exemplar = ops.repeat_elements(self.exemplar, rep=3, axis=0)

        self.penalty = np.ones((config.num_scale)) * config.scale_penalty  # （1，1，1）*0.9745
        self.penalty[config.num_scale // 2] = 1

        # create cosine window
        self.interp_response_sz = config.response_up_stride * config.response_sz
        self.cosine_window = self._cosine_window((self.interp_response_sz,
                                                  self.interp_response_sz))
        self.scales = config.scale_step**np.arange(np.ceil(3/2)-3,
                                                   np.floor(3/2)+1)  # [0.96385542,1,1.0375]

        # create s_x
        self.s_x = s_z + (config.instance_size - config.exemplar_size) / scale_z

        # arbitrary scale saturation
        self.min_s_x = 0.2 * self.s_x
        self.max_s_x = 5 * self.s_x

    def update(self, frame):
        """
         track object based on the previous frame
         Args:
         frame: an RGB image

         Returns:
         bbox: tuple of 1-based bounding box(xmin, ymin, xmax, ymax)
         """
        size_x_scales = self.s_x * self.scales
        pyramid = get_pyramid_instance_image(frame, self.pos, config.instance_size,
                                             size_x_scales, self.img_mean)

        x_crops_tensor = ()
        for k in range(3):
            tmp_x_crop = Tensor(pyramid[k], dtype=ms_np.float32)
            tmp_x_crop = ops.transpose(tmp_x_crop, (2, 0, 1))[np.newaxis, :, :, :]
            x_crops_tensor = x_crops_tensor+(tmp_x_crop,)
        instance_imgs = ms_np.concatenate(x_crops_tensor, axis=0)

        response_maps = self.network(self.exemplar, instance_imgs)
        response_maps = ms_np.squeeze(response_maps)
        response_maps = Tensor.asnumpy(response_maps)
        response_maps_up = [cv2.resize(x, (self.interp_response_sz, self.interp_response_sz),
                                       cv2.INTER_CUBIC)for x in response_maps]
        # get max score
        max_score = np.array([x.max() for x in response_maps_up]) * self.penalty
        # penalty scale change
        scale_idx = max_score.argmax()
        response_map = response_maps_up[scale_idx]
        response_map -= response_map.min()
        response_map /= response_map.sum()
        response_map = (1 - config.window_influence) * response_map + \
                       config.window_influence * self.cosine_window
        max_r, max_c = np.unravel_index(response_map.argmax(), response_map.shape)
        # displacement in interpolation response
        disp_response_interp = np.array([max_c, max_r]) - (self.interp_response_sz - 1) / 2.
        # displacement in input
        disp_response_input = disp_response_interp*config.total_stride/config.response_up_stride
        # displacement in frame
        scale = self.scales[scale_idx]
        disp_response_frame = disp_response_input*(self.s_x * scale)/config.instance_size
        # position in frame coordinates
        self.pos += disp_response_frame
        # scale damping and saturation
        self.s_x *= ((1 - config.scale_lr) + config.scale_lr * scale)
        self.s_x = max(self.min_s_x, min(self.max_s_x, self.s_x))
        self.target_sz = ((1 - config.scale_lr) + config.scale_lr * scale) * self.target_sz
        box = np.array([
            self.pos[0] + 1 - (self.target_sz[0]) / 2,
            self.pos[1] + 1 - (self.target_sz[1]) / 2,
            self.target_sz[0], self.target_sz[1]])
        return box

    def track(self, img_files, box, visualize=False):
        """
            To get the update track box and calculate time
            Args :
                img_files : the location of img
                box : the first image box, include x, y, width, high
        """
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box  # x，y, w, h
        times = np.zeros(frame_num)

        for f, img_file in tqdm(enumerate(img_files), total=len(img_files)):
            img = cv2.imread(img_file, cv2.IMREAD_COLOR)
            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin
            if visualize:
                show_image(img, boxes[f, :])
        return boxes, times
