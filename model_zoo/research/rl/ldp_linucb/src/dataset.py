# Copyright 2020 Huawei Technologies Co., Ltd
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
MovieLens Environment.
"""

import random
import numpy as np
from mindspore import Tensor

_MAX_NUM_ACTIONS = 1682
_NUM_USERS = 943


def load_movielens_data(data_file):
    """Loads the movielens data and returns the ratings matrix."""
    ratings_matrix = np.zeros([_NUM_USERS, _MAX_NUM_ACTIONS])
    with open(data_file, 'r') as f:
        for line in f.readlines():
            row_infos = line.strip().split()
            user_id = int(row_infos[0])
            item_id = int(row_infos[1])
            rating = float(row_infos[2])
            ratings_matrix[user_id - 1, item_id - 1] = rating
    return ratings_matrix


class MovieLensEnv:
    """
    MovieLens dataset environment for bandit algorithms.

    Args:
        data_file(str): path of movielens file, e.g. 'ua.base'.
        num_movies(int): number of movies for choices.
        rank_k(int): the dim of feature.

    Returns:
        Environment for bandit algorithms.
    """
    def __init__(self, data_file, num_movies, rank_k):
        # Initialization
        self._num_actions = num_movies
        self._context_dim = rank_k

        # Load Movielens dataset
        self._data_matrix = load_movielens_data(data_file)
        # Keep only the first items
        self._data_matrix = self._data_matrix[:, :num_movies]
        # Filter the users with at least one rating score
        nonzero_users = list(
            np.nonzero(
                np.sum(
                    self._data_matrix,
                    axis=1) > 0.0)[0])
        self._data_matrix = self._data_matrix[nonzero_users, :]
        # Normalize the data_matrix into -1~1
        self._data_matrix = 0.4 * (self._data_matrix - 2.5)

        # Compute the SVD # Only keep the largest rank_k singular values
        u, s, vh = np.linalg.svd(self._data_matrix, full_matrices=False)
        u_hat = u[:, :rank_k] * np.sqrt(s[:rank_k])
        v_hat = np.transpose(np.transpose(
            vh[:rank_k, :]) * np.sqrt(s[:rank_k]))
        self._approx_ratings_matrix = np.matmul(
            u_hat, v_hat).astype(np.float32)

        # Prepare feature for user i and item j: u[i,:] * vh[:,j]
        # (elementwise product of user feature and item feature)
        self._ground_truth = s
        self._current_user = 0
        self._feature = np.expand_dims(u[:, :rank_k], axis=1) * \
            np.expand_dims(np.transpose(vh[:rank_k, :]), axis=0)
        self._feature = self._feature.astype(np.float32)

    @property
    def ground_truth(self):
        return self._ground_truth

    def observation(self):
        """random select a user and return its feature."""
        sampled_user = random.randint(0, self._data_matrix.shape[0] - 1)
        self._current_user = sampled_user
        return Tensor(self._feature[sampled_user])

    def current_rewards(self):
        """rewards for current user."""
        return Tensor(self._approx_ratings_matrix[self._current_user])
