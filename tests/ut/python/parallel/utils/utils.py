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
from mindspore.common.api import _cell_graph_executor


class ParallelValidator:
    """
    Validator for distribute operator.

    Args:
        net (Cell): `auto_parallel_mode` = True for networks where compile has been executed.

    Examples:
        >>> from mindspore.common.api import _cell_graph_executor
        >>> from parallel.util.utils import ParallelValidator
        >>> net = Net()
        >>> net.set_auto_parallel()
        >>> net.set_train()
        >>> phase, _ = _cell_graph_executor.compile(net, *inputs, auto_parallel_mode=True)
        >>> validator = ParallelValidator(net, phase) # Init validator by net and phase
        >>> assert validator.check_parameter_shape("x", [8, 3, 256, 256]) # Check parameter slice shape
        >>> # expect_layout: (device_arrangement, tensor_map, slice_shape, field_size, uniform_split, opt_shard_group)
        >>> expect_layout = ([4, 2], [1, -1, -1, -1], [8, 3, 256, 256], 0, True, '')
        >>> assert validator.check_parameter_laytout("x", expect_layout)
        >>> # check attrs for "ROIAlign-0" from graph_1
        >>> expect_attrs = {'pooled_height': POOLED_HEIGHT, 'pooled_width': POOLED_WIDTH}
        >>> assert validator.check_node_attrs("ROIAlign-0", expect_attrs, graph_id=1)
        >>> # check node inputs for "ROIAlign-0 from graph_0 (default graph_id)
        >>> expect_inputs = ['features', 'TensorScatterUpdate-0']
        >>> assert validator.check_node_inputs('ROIAlign-0', 'features', 'TensorScatterUpdate-0')
        >>> # check sub graph structure from graph_1
        >>> sub_graph = {
        ...     'ROIAlign-0': ['features', 'TensorScatterUpdate-0'],
        ...     'MaskedFill-0': ['ROIAlign-0', 'ExpandDims-2', 0.0],
        ...     'AllReduce-0': ['MaskedFill-0']
        ... }
        >>> assert validator.check_graph_structure(sub_graph, graph_id=1)

    """
    def __init__(self, net, phase):
        self._parameter_layout_dict = net.parameter_layout_dict
        self._graph_info_dict = _cell_graph_executor._graph_executor.get_parallel_graph_info(phase)

    @property
    def parameter_layout_dict(self):
        return self._parameter_layout_dict

    @property
    def graph_info_dict(self):
        return self._graph_info_dict

    def check_parameter_layout(self, param_name: str, layout: [tuple, list]) -> bool:
        """Verify parameter layout."""
        if not isinstance(layout, (tuple, list)):
            raise TypeError("Type of expect_inputs must be list or tuple, but got {}".format(type(layout)))

        if param_name not in self._parameter_layout_dict.keys():
            return False
        return self._parameter_layout_dict[param_name] == layout

    def check_parameter_shape(self, param_name: str, shape: [tuple, list]) -> bool:
        """Verify parameter shape"""
        if not isinstance(shape, (tuple, list)):
            raise TypeError("Type of expect_inputs must be list or tuple, but got {}".format(type(shape)))

        if param_name not in self._parameter_layout_dict.keys():
            return False
        return self._parameter_layout_dict[param_name][2] == shape

    def check_node_attrs(self, node_name: str, expect_attrs: dict, graph_id=0) -> bool:
        if not isinstance(expect_attrs, dict):
            raise TypeError("Type of expect_attrs must be dict, but got {}".format(type(expect_attrs)))

        cnode_info_dict = self._get_graph_cnode_info(graph_id)
        if node_name not in cnode_info_dict.keys():
            return False
        attrs = cnode_info_dict[node_name]['attrs']
        for attr_name in expect_attrs.keys():
            if attr_name not in attrs.keys() or attrs[attr_name] != expect_attrs[attr_name]:
                return False
        return True

    def check_node_inputs(self, node_name: str, expect_inputs: [tuple, list], graph_id=0) -> bool:
        if not isinstance(expect_inputs, (tuple, list)):
            raise TypeError("Type of expect_inputs must be list or tuple, but got {}".format(type(expect_inputs)))

        cnode_info_dict = self._get_graph_cnode_info(graph_id)
        expect_inputs = list(expect_inputs)
        if node_name not in cnode_info_dict.keys():
            return False
        inputs = cnode_info_dict[node_name]['inputs']
        return inputs == expect_inputs

    def check_graph_structure(self, nodes_dict: dict, graph_id=0) -> bool:
        if not isinstance(nodes_dict, dict):
            raise TypeError("Type of nodes_dict must be dict, but got {}".format(type(nodes_dict)))
        for name, inputs in nodes_dict.items():
            if not self.check_node_inputs(name, inputs, graph_id):
                return False
        return True

    def _get_graph_cnode_info(self, graph_id):
        graph_name = "@graph_" + str(graph_id)
        if graph_name not in self._graph_info_dict.keys():
            raise ValueError("{} is not exist".format(graph_name))
        return self._graph_info_dict[graph_name]
