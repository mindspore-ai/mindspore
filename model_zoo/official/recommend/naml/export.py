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
import numpy as np
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint, export
from src.naml import NAML, NAMLWithLossCell
from src.option import get_args

if __name__ == '__main__':

    args = get_args("export")
    net = NAML(args)
    net.set_train(False)
    net_with_loss = NAMLWithLossCell(net)
    load_checkpoint(args.checkpoint_path, net_with_loss)
    news_encoder = net.news_encoder
    user_encoder = net.user_encoder
    bs = args.batch_size
    category = Tensor(np.zeros([bs, 1], np.int32))
    subcategory = Tensor(np.zeros([bs, 1], np.int32))
    title = Tensor(np.zeros([bs, args.n_words_title], np.int32))
    abstract = Tensor(np.zeros([bs, args.n_words_abstract], np.int32))

    news_input_data = [category, subcategory, title, abstract]
    export(news_encoder, *news_input_data, file_name=f"naml_news_encoder_bs_{bs}", file_format=args.file_format)

    browsed_news = Tensor(np.zeros([bs, args.n_browsed_news, args.n_filters], np.float32))
    export(user_encoder, browsed_news, file_name=f"naml_user_encoder_bs_{bs}", file_format=args.file_format)
