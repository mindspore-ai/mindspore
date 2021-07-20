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
"""run fairmot."""
import os
import os.path as osp
from src.backbone_dla_conv import DLASegConv
from src.opts import Opts
from src.infer_net import InferNet
from src.fairmot_pose import WithNetCell
from src.tracking_utils.utils import mkdir_if_missing
from src.tracking_utils.log import logger
import src.utils.jde as datasets
import fairmot_eval
from mindspore.train.serialization import load_checkpoint


def export(opt):
    """run fairmot."""
    result_root = opt.output_root if opt.output_root != '' else '.'
    mkdir_if_missing(result_root)

    logger.info('Starting tracking...')
    dataloader = datasets.LoadVideo(opt.input_video, opt.img_size)
    result_filename = os.path.join(result_root, 'results.txt')
    frame_rate = dataloader.frame_rate
    frame_dir = None if opt.output_format == 'text' else osp.join(result_root, 'frame')
    backbone_net = DLASegConv(opt.heads,
                              down_ratio=4,
                              final_kernel=1,
                              last_level=5,
                              head_conv=256,
                              is_training=True)
    load_checkpoint(opt.load_model, net=backbone_net)
    infer_net = InferNet()
    net = WithNetCell(backbone_net, infer_net)
    net.set_train(False)
    fairmot_eval.eval_seq(opt, net, dataloader, 'mot', result_filename,
                          save_dir=frame_dir, show_image=False, frame_rate=frame_rate)
    if opt.output_format == 'video':
        output_video_path = osp.join(result_root, 'MOT16-03-results.mp4')
        cmd_str = 'ffmpeg -f image2 -i {}/%05d.jpg -b 5000k -c:v mpeg4 {}' \
            .format(osp.join(result_root, 'frame'),
                    output_video_path)
        os.system(cmd_str)


if __name__ == '__main__':
    opts = Opts().init()
    export(opts)
