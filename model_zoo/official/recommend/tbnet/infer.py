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
"""TBNet inference."""

import os
import argparse

from mindspore import load_checkpoint, load_param_into_net, context
from src.config import TBNetConfig
from src.tbnet import TBNet
from src.aggregator import InferenceAggregator
from src import dataset
from src import steam


def get_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description='Infer TBNet.')

    parser.add_argument(
        '--dataset',
        type=str,
        required=False,
        default='steam',
        help="'steam' dataset is supported currently"
    )

    parser.add_argument(
        '--csv',
        type=str,
        required=False,
        default='infer.csv',
        help="the csv datafile inside the dataset folder (e.g. infer.csv)"
    )

    parser.add_argument(
        '--checkpoint_id',
        type=int,
        required=True,
        help="use which checkpoint(.ckpt) file to infer"
    )

    parser.add_argument(
        '--user',
        type=int,
        required=True,
        help="id of the user to be recommended to"
    )

    parser.add_argument(
        '--items',
        type=int,
        required=False,
        default=1,
        help="no. of items to be recommended"
    )

    parser.add_argument(
        '--explanations',
        type=int,
        required=False,
        default=3,
        help="no. of recommendation explanations to be shown"
    )

    parser.add_argument(
        '--device_id',
        type=int,
        required=False,
        default=0,
        help="device id"
    )

    parser.add_argument(
        '--device_target',
        type=str,
        required=False,
        default='GPU',
        choices=['GPU'],
        help="run code on GPU"
    )

    parser.add_argument(
        '--run_mode',
        type=str,
        required=False,
        default='graph',
        choices=['graph', 'pynative'],
        help="run code by GRAPH mode or PYNATIVE mode"
    )

    return parser.parse_args()


def infer_tbnet():
    """Inference process."""
    args = get_args()
    context.set_context(device_id=args.device_id)
    if args.run_mode == 'graph':
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)

    home = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(home, 'data', args.dataset, 'config.json')
    translate_path = os.path.join(home, 'data', args.dataset, 'translate.json')
    data_path = os.path.join(home, 'data', args.dataset, args.csv)
    ckpt_path = os.path.join(home, 'checkpoints')

    print(f"creating TBNet from checkpoint {args.checkpoint_id}...")
    config = TBNetConfig(config_path)
    network = TBNet(config)
    param_dict = load_checkpoint(os.path.join(ckpt_path, f'tbnet_epoch{args.checkpoint_id}.ckpt'))
    load_param_into_net(network, param_dict)

    print(f"creating dataset from {data_path}...")
    infer_ds = dataset.create(data_path, config.per_item_num_paths, train=False, users=args.user)
    infer_ds = infer_ds.batch(config.batch_size)

    print("inferring...")
    # infer and aggregate results
    aggregator = InferenceAggregator(top_k=args.items)
    for user, item, relation1, entity, relation2, hist_item, rating in infer_ds:
        del rating
        result = network(item, relation1, entity, relation2, hist_item)
        item_score = result[0]
        path_importance = result[1]
        aggregator.aggregate(user, item, relation1, entity, relation2, hist_item, item_score, path_importance)

    # show recommendations with explanations
    explainer = steam.TextExplainer(translate_path)
    recomms = aggregator.recommend()
    for user, recomm in recomms.items():
        for item_rec in recomm.item_records:

            item_name = explainer.translate_item(item_rec.item)
            print(f"Recommend <{item_name}> to user:{user}, because:")

            # show explanations
            explanation = 0
            for path in item_rec.paths:
                print(" - " + explainer.explain(path))
                explanation += 1
                if explanation >= args.explanations:
                    break
            print("")


if __name__ == '__main__':
    infer_tbnet()
