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
"""Evaluation NAML."""
from mindspore.common import set_seed
from mindspore.train.serialization import load_checkpoint
from src.naml import NAML, NAMLWithLossCell
from src.option import get_args
from src.dataset import MINDPreprocess
from src.utils import NAMLMetric, get_metric

if __name__ == '__main__':
    args = get_args("eval")
    set_seed(args.seed)
    net = NAML(args)
    net.set_train(False)
    net_with_loss = NAMLWithLossCell(net)
    load_checkpoint(args.checkpoint_path, net_with_loss)
    news_encoder = net.news_encoder
    user_encoder = net.user_encoder
    metric = NAMLMetric()
    mindpreprocess = MINDPreprocess(vars(args), dataset_path=args.eval_dataset_path)
    get_metric(args, mindpreprocess, news_encoder, user_encoder, metric)
