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
"""maskrcnn testing script."""

import os
import pytest
import numpy as np
from model_zoo.official.cv.maskrcnn.src.maskrcnn.mask_rcnn_r50 import Mask_Rcnn_Resnet50
from model_zoo.official.cv.maskrcnn.src.config import config

from mindspore import Tensor, context, export

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_maskrcnn_export():
    """
    export maskrcnn air.
    """
    net = Mask_Rcnn_Resnet50(config=config)
    net.set_train(False)

    bs = config.test_batch_size

    img = Tensor(np.zeros([bs, 3, 768, 1280], np.float16))
    img_metas = Tensor(np.zeros([bs, 4], np.float16))
    gt_bboxes = Tensor(np.zeros([bs, 128, 4], np.float16))
    gt_labels = Tensor(np.zeros([bs, 128], np.int32))
    gt_num = Tensor(np.zeros([bs, 128], np.bool))
    gt_mask = Tensor(np.zeros([bs, 128], np.bool))

    input_data = [img, img_metas, gt_bboxes, gt_labels, gt_num, gt_mask]
    export(net, *input_data, file_name="maskrcnn", file_format="AIR")
    file_name = "maskrcnn.air"
    assert os.path.exists(file_name)
    os.remove(file_name)


if __name__ == '__main__':
    test_maskrcnn_export()
