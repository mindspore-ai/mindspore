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
"""Three-dimension embedding vector initialization."""

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore._checkparam import Rel
from mindspore._checkparam import Validator as validator
from mindspore.nn import Cell


class EmbeddingMatrix(Cell):
    """
    Support three-dimension embedding vector initialization.
    """

    def __init__(self, vocab_size, embedding_size, embedding_table='normal',
                 dtype=mstype.float32, padding_idx=None):
        super(EmbeddingMatrix, self).__init__()
        self.vocab_size = validator.check_value_type('vocab_size', vocab_size, [int], self.cls_name)
        self.embedding_size = validator.check_value_type('embedding_size', embedding_size,
                                                         [int, tuple, list], self.cls_name)
        validator.check_subclass("dtype", dtype, mstype.number_type, self.cls_name)
        self.dtype = dtype
        if isinstance(self.embedding_size, int):
            self.init_tensor = initializer(embedding_table, [vocab_size, embedding_size])
            self.embedding_out = (self.embedding_size,)
        else:
            if len(self.embedding_size) != 2:
                raise ValueError("embedding_size should be a int or a tuple of two ints")
            self.init_tensor = initializer(embedding_table, [vocab_size, self.embedding_size[0],
                                                             self.embedding_size[1]])
            self.embedding_out = (self.embedding_size[0], self.embedding_size[1],)
        self.padding_idx = padding_idx
        if padding_idx is not None:
            self.padding_idx = validator.check_int_range(padding_idx, 0, vocab_size, Rel.INC_BOTH,
                                                         "padding_idx", self.cls_name)
            if isinstance(self.init_tensor, Tensor) and self.init_tensor.init is not None:
                self.init_tensor = self.init_tensor.init_data()
            self.init_tensor = self.init_tensor.asnumpy()
            self.init_tensor[self.padding_idx] = 0
            self.init_tensor = Tensor(self.init_tensor)
        self.embedding_table = Parameter(self.init_tensor, name='embedding_table')
        self.expand = P.ExpandDims()
        self.reshape_flat = P.Reshape()
        self.shp_flat = (-1,)
        self.gather = P.Gather()
        self.reshape = P.Reshape()
        self.get_shp = P.Shape()

    def construct(self, ids):
        """
        Return the initialized three-dimension embedding vector
        """
        extended_ids = self.expand(ids, -1)
        out_shape = self.get_shp(ids) + self.embedding_out
        flat_ids = self.reshape_flat(extended_ids, self.shp_flat)
        output_for_reshape = self.gather(self.embedding_table, flat_ids, 0)
        output = self.reshape(output_for_reshape, out_shape)
        return output
