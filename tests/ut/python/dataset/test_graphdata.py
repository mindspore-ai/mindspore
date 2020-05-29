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
# ==============================================================================
import pytest
import numpy as np
import mindspore.dataset as ds
from mindspore import log as logger

DATASET_FILE = "../data/mindrecord/testGraphData/testdata"


def test_graphdata_getfullneighbor():
    g = ds.GraphData(DATASET_FILE, 2)
    nodes = g.get_all_nodes(1)
    assert len(nodes) == 10
    neighbor = g.get_all_neighbors(nodes, 2)
    assert neighbor.shape == (10, 6)
    row_tensor = g.get_node_feature(neighbor.tolist(), [2, 3])
    assert row_tensor[0].shape == (10, 6)


def test_graphdata_getnodefeature_input_check():
    g = ds.GraphData(DATASET_FILE)
    with pytest.raises(TypeError):
        input_list = [1, [1, 1]]
        g.get_node_feature(input_list, [1])

    with pytest.raises(TypeError):
        input_list = [[1, 1], 1]
        g.get_node_feature(input_list, [1])

    with pytest.raises(TypeError):
        input_list = [[1, 1], [1, 1, 1]]
        g.get_node_feature(input_list, [1])

    with pytest.raises(TypeError):
        input_list = [[1, 1, 1], [1, 1]]
        g.get_node_feature(input_list, [1])

    with pytest.raises(TypeError):
        input_list = [[1, 1], [1, [1, 1]]]
        g.get_node_feature(input_list, [1])

    with pytest.raises(TypeError):
        input_list = [[1, 1], [[1, 1], 1]]
        g.get_node_feature(input_list, [1])

    with pytest.raises(TypeError):
        input_list = [[1, 1], [1, 1]]
        g.get_node_feature(input_list, 1)

    with pytest.raises(TypeError):
        input_list = [[1, 0.1], [1, 1]]
        g.get_node_feature(input_list, 1)

    with pytest.raises(TypeError):
        input_list = np.array([[1, 0.1], [1, 1]])
        g.get_node_feature(input_list, 1)

    with pytest.raises(TypeError):
        input_list = [[1, 1], [1, 1]]
        g.get_node_feature(input_list, ["a"])

    with pytest.raises(TypeError):
        input_list = [[1, 1], [1, 1]]
        g.get_node_feature(input_list, [1, "a"])


if __name__ == '__main__':
    test_graphdata_getfullneighbor()
    logger.info('test_graphdata_getfullneighbor Ended.\n')
    test_graphdata_getnodefeature_input_check()
    logger.info('test_graphdata_getnodefeature_input_check Ended.\n')
