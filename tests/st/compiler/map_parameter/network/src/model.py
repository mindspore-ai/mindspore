# Copyright 2022 Huawei Technologies Co., Ltd
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

import copy

import mindspore as ms
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore import _checkparam as validator
from mindspore.ops.primitive import constexpr
from mindspore.nn.layer.basic import ClipByNorm
from mindspore.experimental import MapParameter

from mindspore.nn import Cell, Flatten, Dense
from mindspore.nn import SoftmaxCrossEntropyWithLogits
from mindspore.nn import Adam
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint
from mindspore.train import Accuracy
from mindspore.common import set_seed


@constexpr
def _make_axis_range(start, end):
    axis = tuple(range(start, end))
    return axis


class HashEmbeddingLookup(Cell):
    def __init__(self, embedding_size, key_dtype=ms.int32, param_init='normal', max_norm=None, sparse=True):
        """Initialize HashEmbeddingLookup."""
        super(HashEmbeddingLookup, self).__init__()
        validator.check_value_type('sparse', sparse, [bool], self.cls_name)

        self.forward_unique = sparse
        self.embedding_size = validator.check_positive_int(embedding_size, 'embedding_size', self.cls_name)
        self.embedding_table = MapParameter(key_dtype=key_dtype, value_dtype=ms.float32, value_shape=(embedding_size,),
                                            default_value=param_init, name='embedding_table')

        # Ops for sparse mode.
        # pylint: disable=W0212
        self.map_tensor_get = P._map_tensor_ops.MapTensorGet(True)
        self.gather_revert = P.Gather()
        self.reshape_first = P.Reshape()
        self.reshape = P.Reshape()
        self.unique = P.Unique()
        self.shape = P.Shape()

        self.embedding_table.unique = self.forward_unique

        self.max_norm = max_norm
        if self.max_norm is not None:
            self.max_norm = validator.check_positive_float(self.max_norm, 'max_norm', self.cls_name)
            self.max_norm = Tensor(self.max_norm, dtype=mstype.float32)


    def construct(self, indices):
        if self.forward_unique:
            shp = self.shape(indices) + (self.embedding_size,)
            indices_flatten = self.reshape_first(indices, (-1,))
            unique_id, unique_idx = self.unique(indices_flatten)
            weight_unique = self.map_tensor_get(self.embedding_table, unique_id)
            weight_flatten = self.gather_revert(weight_unique, unique_idx, 0)
            out = self.reshape(weight_flatten, shp)
        else:
            out = self.embedding_table.get(indices)

        if self.max_norm is not None:
            axis = _make_axis_range(F.rank(indices), F.rank(out))
            clip_by_norm = ClipByNorm(axis)
            out = clip_by_norm(out, self.max_norm)
        return out


class Net(Cell):
    def __init__(self, in_channels, out_channels, embedding_size, sparse):
        super().__init__()
        set_seed(5)
        self.embedding_lookup1 = HashEmbeddingLookup(embedding_size=embedding_size, param_init='normal',
                                                     sparse=sparse)
        self.flatten = Flatten()
        self.dense = Dense(in_channels=in_channels, out_channels=out_channels, weight_init='normal',
                           has_bias=False)
        self.type = ms.int32
        self.cast = P.Cast()

    def construct(self, x):
        x = self.flatten(x)
        x = self.cast(x, self.type)
        x = self.embedding_lookup1(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.flatten(x)
        return x


class ModelExecutor:
    def __init__(self, dataset, input_shape, in_channels=320, out_channels=3,
                 embedding_size=10, epoch_size=2, sparse=True, save_ckpt=False):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.embedding_size = embedding_size
        self.train_dataset = dataset
        self.eval_dataset = copy.deepcopy(dataset)
        self.epoch_size = epoch_size
        self.sparse = sparse
        self.save_ckpt = save_ckpt

    def run_dynamic_embedding(self):
        net = Net(self.in_channels, self.out_channels, self.embedding_size, self.sparse)
        net.set_train()
        loss = SoftmaxCrossEntropyWithLogits(reduction='mean')
        # pylint: disable=E1123
        opt = Adam(params=filter(lambda x: x.requires_grad, net.get_parameters()), use_lazy=True)

        model = Model(net, loss, opt, metrics={"Accuracy": Accuracy()})
        callback_list = []
        if self.save_ckpt:
            config = CheckpointConfig(save_checkpoint_steps=1, keep_checkpoint_max=1)
            ckpoint_cb = ModelCheckpoint(prefix="ckpt_dynamic_embedding", directory='./ckpt',
                                         config=config)
            callback_list.append(ckpoint_cb)
        model.train(self.epoch_size, self.train_dataset, callbacks=callback_list, dataset_sink_mode=True)
        acc = model.eval(self.eval_dataset, dataset_sink_mode=True)
        return acc['Accuracy']
