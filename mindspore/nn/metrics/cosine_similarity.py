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
import numpy as np
from mindspore._checkparam import Validator as validator
from .metric import Metric


class CosineSimilarity(Metric):
    """
    Computes representation similarity

    Args:
        similarity (str): 'dot' or 'cosine'. Default: 'cosine'
        reduction (str): 'none', 'sum', 'mean' (all along dim -1). Default: 'none'
        zero_diagonal (bool): if True, the diagonals are set to zero. Default: True

    Return:
        A square matrix (input1, input1) with the similarity scores between all elements.
        If sum or mean are used, then returns (b, 1) with the reduced value for each row

    Example:
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
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self.sqr_mtx_res = 0
        self._is_update = False

    def update(self, inputs):
        """
        Updates the internal evaluation result with 'input1'.

        Args:
            inputs: input_data `input1`. The input_data is a `Tensor` or an array.
        """
        input_data = self._convert_data(inputs)

        if self.similarity == 'cosine':
            data = np.linalg.norm(input_data, ord=2, axis=1)
            input_data = input_data / np.expand_dims(data, 1)

        self.sqr_mtx_res = np.dot(input_data, input_data.transpose(1, 0))
        self._is_update = True

    def eval(self):
        """
         Computes the Cosine_Similarity square matrix.

         Returns:
             A square matrix.

        """
        if not self._is_update:
            raise RuntimeError('Call the update method before calling eval.')

        if self.zero_diagonal:
            np.fill_diagonal(self.sqr_mtx_res, 0)

        if self.reduction == 'mean':
            self.sqr_mtx_res = np.mean(self.sqr_mtx_res, axis=-1)

        if self.reduction == 'sum':
            self.sqr_mtx_res = np.sum(self.sqr_mtx_res, axis=-1)

        return self.sqr_mtx_res
