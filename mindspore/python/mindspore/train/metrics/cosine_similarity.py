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
"""CosineSimilarity."""
from __future__ import absolute_import

import numpy as np

from mindspore import _checkparam as validator
from mindspore.train.metrics.metric import Metric, rearrange_inputs


class CosineSimilarity(Metric):
    """
    Computes representation similarity.

    Args:
        similarity (str): 'dot' or 'cosine'. Default: ``'cosine'`` .
        reduction (str): ``'none'``, 'sum', ``'mean'`` (all along dim -1). Default: ``'none'`` .
        zero_diagonal (bool): If True,  diagonals of results will be set to zero. Default: ``True`` .

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.train import CosineSimilarity
        >>>
        >>> test_data = np.array([[1, 3, 4, 7], [2, 4, 2, 5], [3, 1, 5, 8]])
        >>> metric = CosineSimilarity()
        >>> metric.clear()
        >>> metric.update(test_data)
        >>> square_matrix = metric.eval()
        >>> print(square_matrix)
        [[0.  0.94025615  0.95162452]
         [0.94025615  0.  0.86146098]
         [0.95162452  0.86146098  0.]]
    """
    def __init__(self, similarity='cosine', reduction='none', zero_diagonal=True):
        super().__init__()
        similarity_list = ['dot', 'cosine']
        reduction_list = ['none', 'sum', 'mean']
        similarity = validator.check_value_type("similarity", similarity, [str])
        self.similarity = validator.check_string(similarity, similarity_list, "similarity")
        reduction = validator.check_value_type("reduction", reduction, [str])
        self.reduction = validator.check_string(reduction, reduction_list, "reduction")
        self.zero_diagonal = validator.check_value_type("zero_diagonal", zero_diagonal, [bool])
        self.sqr_mtx_res = 0
        self.clear()
        self._is_update = None

    def clear(self):
        """Clears the internal evaluation result."""
        self.sqr_mtx_res = 0
        self._is_update = False

    @rearrange_inputs
    def update(self, inputs):
        """
        Updates the internal evaluation result with 'inputs'.

        Args:
            inputs (Union[Tensor, list, numpy.ndarray]): The input matrix.
        """
        input_data = self._convert_data(inputs)

        if self.similarity == 'cosine':
            data = np.linalg.norm(input_data, ord=2, axis=1)
            input_data = input_data / np.expand_dims(data, 1)

        self.sqr_mtx_res = np.dot(input_data, input_data.transpose(1, 0))
        self._is_update = True

    def eval(self):
        """
         Computes the similarity matrix.

         Returns:
             numpy.ndarray. The similarity matrix.

         Raises:
            RuntimeError: If the update method is not called first, an error will be reported.
        """
        if not self._is_update:
            raise RuntimeError("Please call the 'update' method before calling 'eval' method.")

        if self.zero_diagonal:
            np.fill_diagonal(self.sqr_mtx_res, 0)

        if self.reduction == 'mean':
            self.sqr_mtx_res = np.mean(self.sqr_mtx_res, axis=-1)

        if self.reduction == 'sum':
            self.sqr_mtx_res = np.sum(self.sqr_mtx_res, axis=-1)

        return self.sqr_mtx_res
