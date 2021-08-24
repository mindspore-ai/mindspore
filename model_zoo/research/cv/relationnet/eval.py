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
"""eval"""
import argparse
import os
import random
import numpy as np
from mindspore import context
from mindspore.ops import operations as ops
from mindspore import load_checkpoint, load_param_into_net
from mindspore.communication.management import init
from mindspore.context import ParallelMode
from mindspore.common import dtype as mstype
import src.dataset as dt
from src.config import relationnet_cfg as cfg
from src.relationnet import Encoder_Relation, weight_init
import scipy.stats
import scipy.special as sc


parser = argparse.ArgumentParser(description="One Shot Visual Recognition")
parser.add_argument("-u", "--hidden_unit", type=int, default=10)
parser.add_argument("-dt", "--device_target", type=str, default='Ascend', choices=("Ascend"),
                    help="Device target, support Ascend.")
parser.add_argument("-di", "--device_id", type=int, default=0)
parser.add_argument("--ckpt_dir", default='./ckpt/', help='the path of output')
parser.add_argument("--data_path", default='/data/omniglot_resized/',
                    help="Path where the dataset is saved")
parser.add_argument("--data_url", default=None)
parser.add_argument("--train_url", default=None)
parser.add_argument("--cloud", default=None, help='if run on cloud')
args = parser.parse_args()


context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)


# init operators
concat0dim = ops.Concat(axis=0)


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sc.stdtrit(n-1, (1+confidence)/2.)
    return m, h


def main():
    local_data_url = args.data_path
    local_train_url = args.ckpt_dir
    # if run on the cloud
    if args.cloud:
        import moxing as mox
        local_data_url = './cache/data'
        local_train_url = './cache/ckpt'
        device_target = args.device_target
        device_num = int(os.getenv("RANK_SIZE"))
        device_id = int(os.getenv("DEVICE_ID"))
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
        context.set_context(save_graphs=False)
        if device_target == "Ascend":
            context.set_context(device_id=device_id)

            if device_num > 1:
                context.reset_auto_parallel_context()
                context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                                  gradients_mean=True)
                init()
                local_data_url = os.path.join(local_data_url, str(device_id))
        else:
            raise ValueError("Unsupported platform.")
        import moxing as mox
        mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)
        mox.file.copy_parallel(src_url=args.ckpt_dir, dst_url=local_train_url)
    else:
        # run on the local server
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, device_id=args.device_id)
        context.set_context(save_graphs=False)

    # Step 1: init data folders
    print("init data folders")
    _, metatest_character_folders = dt.omniglot_character_folders(data_path=local_data_url)

    #Step 4 : init networks
    print("init neural networks")
    encoder_relation = Encoder_Relation(cfg.feature_dim, cfg.relation_dim)
    encoder_relation.set_train(False)
    weight_init(encoder_relation)

    #load parameters
    if os.path.exists(local_train_url):
        param_dict = load_checkpoint(local_train_url)
        load_param_into_net(encoder_relation, param_dict)
        print("successfully load parameters")
    else:
        print("Error:can not load checkpoint")

    total_accuracy = 0.0
    print("=" * 10 + "Testing" + "=" * 10)
    for episode in range(cfg.eval_episode):
        total_rewards = 0
        accuracies = []
        for _ in range(cfg.test_episode):
            degrees = random.choice([0, 90, 180, 270])
            flip = random.choice([True, False])
            task = dt.OmniglotTask(metatest_character_folders, cfg.class_num, cfg.sample_num_per_class,
                                   cfg.sample_num_per_class)
            sample_dataloader = dt.get_data_loader(task, num_per_class=cfg.sample_num_per_class, split="train",
                                                   shuffle=False, rotation=degrees, flip=flip)
            test_dataloader = dt.get_data_loader(task, num_per_class=cfg.sample_num_per_class, split="test",
                                                 shuffle=True, rotation=degrees, flip=flip)
            test_samples, _ = next(sample_dataloader)
            test_batches, test_batch_labels = next(test_dataloader)

            # concat samples and batches
            test_input = concat0dim((test_samples, test_batches))
            test_relations = encoder_relation(test_input)

            predict_labels = ops.Argmax(axis=1, output_type=mstype.int32)(test_relations).asnumpy()
            test_batch_labels = test_batch_labels.asnumpy().astype(np.int32)
            rewards = [1 if predict_labels[j] == test_batch_labels[j] else 0 for j in range(cfg.class_num)]
            total_rewards += np.sum(rewards)
            accuracy = np.sum(rewards) / 1.0 / cfg.class_num / cfg.sample_num_per_class
            accuracies.append(accuracy)

        test_accuracy, h = mean_confidence_interval(accuracies)
        total_accuracy += test_accuracy
        print('-' * 5 + 'Episode {}/{}'.format(episode + 1, cfg.eval_episode) + '-' * 5)
        print("test accuracy: %.4f or %.4f%%  h: %f" % (test_accuracy, test_accuracy*100, h))

    print("aver_accuracy : %.2f" % (total_accuracy/cfg.eval_episode*100))


if __name__ == '__main__':
    main()
