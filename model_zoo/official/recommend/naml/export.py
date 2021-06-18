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
"""NAML export."""
import os
import numpy as np
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, export
from src.naml import NAML, NAMLWithLossCell
from model_utils.config import config
from model_utils.moxing_adapter import moxing_wrapper
from model_utils.device_adapter import get_device_id


def modelarts_pre_process():
    '''modelarts pre process function.'''
    config.export_file_dir = os.path.join(config.output_path, config.export_file_dir)


@moxing_wrapper(pre_process=modelarts_pre_process)
def run_export():
    """run export."""
    config.phase = "export"
    config.device_id = get_device_id()
    config.neg_sample = config.export_neg_sample
    context.set_context(mode=context.GRAPH_MODE, device_target=config.platform, device_id=config.device_id,
                        save_graphs=config.save_graphs, save_graphs_path="naml_ir")

    net = NAML(config)
    net.set_train(False)
    net_with_loss = NAMLWithLossCell(net)
    load_checkpoint(config.checkpoint_path, net_with_loss)
    news_encoder = net.news_encoder
    user_encoder = net.user_encoder
    bs = config.batch_size
    category = Tensor(np.zeros([bs, 1], np.int32))
    subcategory = Tensor(np.zeros([bs, 1], np.int32))
    title = Tensor(np.zeros([bs, config.n_words_title], np.int32))
    abstract = Tensor(np.zeros([bs, config.n_words_abstract], np.int32))

    news_input_data = [category, subcategory, title, abstract]
    export(news_encoder, *news_input_data,
           file_name=os.path.join(config.export_file_dir, f"naml_news_encoder_bs_{bs}"),
           file_format=config.file_format)

    browsed_news = Tensor(np.zeros([bs, config.n_browsed_news, config.n_filters], np.float32))
    export(user_encoder, browsed_news,
           file_name=os.path.join(config.export_file_dir, f"naml_user_encoder_bs_{bs}"),
           file_format=config.file_format)


if __name__ == '__main__':
    run_export()
