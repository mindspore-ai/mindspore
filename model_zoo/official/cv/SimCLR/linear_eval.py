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
"""
######################## eval SimCLR example ########################
eval SimCLR according to model file:
python eval.py --encoder_checkpoint_path Your.ckpt --train_dataset_path /YourDataPath1
               --eval_dataset_path /YourDataPath2
"""
import ast
import os
import argparse
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import ops
from mindspore import context
from mindspore.common.initializer import TruncatedNormal
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.common import set_seed
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank
from src.dataset import create_dataset
from src.simclr_model import SimCLR
from src.resnet import resnet50 as resnet
from src.reporter import Reporter
from src.optimizer import get_eval_optimizer as get_optimizer



parser = argparse.ArgumentParser(description='Linear Evaluation Protocol')
parser.add_argument('--device_target', type=str, default='Ascend',
                    help="Device target, Currently only Ascend is supported.")
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Running distributed evaluation.')
parser.add_argument('--run_cloudbrain', type=ast.literal_eval, default=True,
                    help='Whether it is running on CloudBrain platform.')
parser.add_argument('--device_num', type=int, default=1, help='Device num.')
parser.add_argument("--device_id", type=int, default=0, help="device id, default is 0.")
parser.add_argument('--dataset_name', type=str, default="cifar10", help='Dataset, Currently only cifar10 is supported.')
parser.add_argument('--train_url', default=None, help='Cloudbrain Location of training outputs.\
                    This parameter needs to be set when running on the cloud brain platform.')
parser.add_argument('--data_url', default=None, help='Cloudbrain Location of data.\
                    This parameter needs to be set when running on the cloud brain platform.')
parser.add_argument('--train_dataset_path', type=str, default="./cifar/train",\
                    help='Dataset path for training classifier.\
                    This parameter needs to be set when running on the host.')
parser.add_argument('--eval_dataset_path', type=str, default="./cifar/eval",\
                    help='Dataset path for evaluating classifier.\
                    This parameter needs to be set when running on the host.')
parser.add_argument('--train_output_path', type=str, default="./outputs", help='Location of ckpt and log.\
                    This parameter needs to be set when running on the host.')
parser.add_argument("--class_num", type=int, default=10, help="dataset classification number")
parser.add_argument('--batch_size', type=int, default=128, help='batch_size for training classifier, default is 128.')
parser.add_argument('--epoch_size', type=int, default=100, help='epoch size for training classifier, default is 200.')
parser.add_argument('--projection_dimension', type=int, default=128,
                    help='Projection output dimensionality, default is 128.')
parser.add_argument('--width_multiplier', type=int, default=1, help='width_multiplier=4,resnet50x4')
parser.add_argument('--pre_classifier_checkpoint_path', type=str, default=None, help='Classifier Checkpoint file path.')
parser.add_argument('--encoder_checkpoint_path', type=str, default="simclr_156.ckpt",
                    help='Encoder Checkpoint file path.')
parser.add_argument("--save_checkpoint_epochs", type=int, default=10, help="Save checkpoint epochs, default is 1.")
parser.add_argument("--print_iter", type=int, default=100, help="log print iter, default is 100.")
parser.add_argument('--save_graphs', type=ast.literal_eval, default=False,
                    help='whether save graphs, default is False.')
parser.add_argument('--use_norm', type=ast.literal_eval, default=False, help='Dataset normalize.')

args = parser.parse_args()
set_seed(1)
local_data_url = './cache/data'
local_train_url = './cache/train'
_local_train_url = local_train_url

if args.device_target != "Ascend":
    raise ValueError("Unsupported device target.")
if args.run_distribute:
    args.device_id = int(os.getenv("DEVICE_ID"))
    if args.device_num > int(os.getenv("RANK_SIZE")) or args.device_num == 1:
        args.device_num = int(os.getenv("RANK_SIZE"))
    context.set_context(device_id=args.device_id)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=args.save_graphs)
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL,
                                      gradients_mean=True, device_num=args.device_num)
    init()
    args.rank = get_rank()
    local_data_url = os.path.join(local_data_url, str(args.device_id))
    local_train_url = os.path.join(local_train_url, str(args.device_id))
    args.train_output_path = os.path.join(args.train_output_path, str(args.device_id))
else:
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target,
                        save_graphs=args.save_graphs, device_id=args.device_id)
    args.rank = 0
    args.device_num = 1

if args.run_cloudbrain:
    import moxing as mox
    args.train_dataset_path = os.path.join(local_data_url, "train")
    args.eval_dataset_path = os.path.join(local_data_url, "val")
    args.train_output_path = local_train_url
    mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)

class LogisticRegression(nn.Cell):
    """
    Logistic regression
    """
    def __init__(self, n_features, n_classes):
        super(LogisticRegression, self).__init__()
        self.model = nn.Dense(n_features, n_classes, TruncatedNormal(0.02), TruncatedNormal(0.02))

    def construct(self, x):
        x = self.model(x)
        return x

class Linear_Eval():
    """
    Linear classifier
    """
    def __init__(self, net, loss):
        super(Linear_Eval, self).__init__()
        self.net = net
        self.softmax = nn.Softmax()
        self.loss = loss
    def __call__(self, x, y):
        x = self.net(x)
        loss = self.loss(x, y)
        x = self.softmax(x)
        predicts = ops.Argmax(output_type=mstype.int32)(x)
        acc = np.sum(predicts.asnumpy() == y.asnumpy())/len(y.asnumpy())
        return loss.asnumpy(), acc

class Linear_Train(nn.Cell):
    """
    Train linear classifier
    """
    def __init__(self, net, loss, opt):
        super(Linear_Train, self).__init__()
        self.netwithloss = nn.WithLossCell(net, loss)
        self.train_net = nn.TrainOneStepCell(self.netwithloss, opt)
        self.train_net.set_train()
    def construct(self, x, y):
        return self.train_net(x, y)

if __name__ == "__main__":
    base_net = resnet(1, args.width_multiplier, cifar_stem=args.dataset_name == "cifar10")
    simclr_model = SimCLR(base_net, args.projection_dimension, base_net.end_point.in_channels)
    if args.run_cloudbrain:
        mox.file.copy_parallel(src_url=args.encoder_checkpoint_path, dst_url=local_data_url+'/encoder.ckpt')
        simclr_param = load_checkpoint(local_data_url+'/encoder.ckpt')
    else:
        simclr_param = load_checkpoint(args.encoder_checkpoint_path)
    load_param_into_net(simclr_model.encoder, simclr_param)
    classifier = LogisticRegression(simclr_model.n_features, args.class_num)
    dataset = create_dataset(args, dataset_mode="train_classifier")
    optimizer = get_optimizer(classifier, dataset.get_dataset_size(), args)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_Train = Linear_Train(net=classifier, loss=criterion, opt=optimizer)
    reporter = Reporter(args, linear_eval=True)
    reporter.dataset_size = dataset.get_dataset_size()
    reporter.linear_eval = True
    if args.pre_classifier_checkpoint_path:
        if args.run_cloudbrain:
            mox.file.copy_parallel(src_url=args.pre_classifier_checkpoint_path,
                                   dst_url=local_data_url+'/pre_classifier.ckpt')
            classifier_param = load_checkpoint(local_data_url+'/pre_classifier.ckpt')
        else:
            classifier_param = load_checkpoint(args.pre_classifier_checkpoint_path)
        load_param_into_net(classifier, classifier_param)
    else:
        dataset_train = []
        for _, data in enumerate(dataset, start=1):
            _, images, labels = data
            features = simclr_model.inference(images)
            dataset_train.append([features, labels])
        reporter.info('==========start training linear classifier===============')
        # Train.
        for _ in range(args.epoch_size):
            reporter.epoch_start()
            for idx, data in enumerate(dataset_train, start=1):
                features, labels = data
                out = net_Train(features, labels)
                reporter.step_end(out)
            reporter.epoch_end(classifier)
        reporter.info('==========end training  linear classifier===============')

    dataset = create_dataset(args, dataset_mode="eval_classifier")
    reporter.dataset_size = dataset.get_dataset_size()
    net_Eval = Linear_Eval(net=classifier, loss=criterion)
    # Eval.
    reporter.info('==========start evaluating linear classifier===============')
    reporter.start_predict()
    for idx, data in enumerate(dataset, start=1):
        _, images, labels = data
        features = simclr_model.inference(images)
        batch_loss, batch_acc = net_Eval(features, labels)
        reporter.predict_step_end(batch_loss, batch_acc)
    reporter.end_predict()
    reporter.info('==========end evaluating linear classifier===============')
    if args.run_cloudbrain:
        mox.file.copy_parallel(src_url=_local_train_url, dst_url=args.train_url)
