# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Checkpoint strategy info"""
from __future__ import absolute_import

__all__ = ["get_strategy_metadata", "get_current_strategy_metadata", "enable_save_strategy_online", \
           "clear_strategy_metadata"]

from itertools import chain
from typing import Sequence, Union, Tuple, List, Dict
from types import SimpleNamespace

import numpy as np

from mindspore import log as logger
from mindspore._c_expression import StrategyInfo
from mindspore._c_expression import StrategyLayout
from mindspore.parallel.shard import Layout

LayoutInfo = Tuple[Layout, str, str]
StrOrTuple = Union[str, Tuple["StrOrTuple", ...], List["StrOrTuple"]]


def get_strategy_metadata(network, rank_id=None) -> Dict[int, Dict[str, List[LayoutInfo]]]:
    """
    Get all params strategy info or specific rank strategy info in this cell.
    For more information on layouts, please refer to: :class:`mindspore.parallel.Layout`.

    Args:
        network (str): The network name.
        rank_id (int, optional): The rank id of the process on which this cell will be launched.
            Defaults to ``None``, which means strategy metadata for all ranks will be returned.

    Returns:
        Dict. A dictionary containing the parameter slicing strategies for either all ranks or a specific rank.
        The key is `rank_id`, and the value is the slicing strategy for all parameters on that rank.
        Within each rank's strategy, the key is the parameter name, and the value is the slicing strategy.
        If a `rank_id` is specified, the dictionary returns the strategy information for that specific rank.
        Otherwise, it returns the strategy information for all ranks in the network. If not supported, returns None.

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import nn
        >>> from mindspore.communication import init
        >>> from mindspore.nn.utils import no_init_parameters
        >>> from mindspore.parallel.auto_parallel import AutoParallel
        >>> from mindspore.train import Model
        >>> from mindspore.parallel.strategy import get_strategy_metadata, get_current_strategy_metadata,
        ...     enable_save_strategy_online, clear_strategy_metadata
        >>>
        >>> ms.set_context(mode=ms.GRAPH_MODE)
        >>> init()
        >>> ms.set_seed(1)
        >>>
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://atomgit.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> with no_init_parameters():
        ...     net = LeNet5()
        ...     optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>>
        >>> loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        >>> train_net = AutoParallel(net, parallel_mode="semi_auto")
        >>> model = Model(network=train_net, loss_fn=loss, optimizer=optim, metrics=None)
        >>>
        >>> # Create the dataset taking MNIST as an example. Refer to
        >>> # https://atomgit.com/mindspore/docs/blob/master/docs/mindspore/code/mnist.py
        >>> dataset = create_dataset()
        >>>
        >>> enable_save_strategy_online()
        >>> model.train(2, dataset)
        >>>
        >>> global_info = get_strategy_metadata(network=model.train_network)
        >>> rank0_info = get_strategy_metadata(network=model.train_network, rank_id=0)
        >>> local_info = get_current_strategy_metadata(network=model.train_network)
        >>> clear_strategy_metadata()
    """
    return _NetStrategyInfo(network, global_layout=None, local_layout=None).get_rank_layout(rank_id)


def get_current_strategy_metadata(network) -> Dict[int, Dict[str, List[LayoutInfo]]]:
    """
    Get parameters dictionary of cur rank of the network.

    Args:
        network(str): The network name.

    Returns:
        Dict. The key is 0 (representing the local rank), and the value is the slicing strategy for all parameters.
        The key within the value represents the parameter name, and the value is the corresponding slicing strategy \
        for that parameter. If not supported, returns None.
    """
    return _NetStrategyInfo(network, global_layout=None, local_layout=None).get_local_rank_layout()


def enable_save_strategy_online():
    """
    Enable save strategy metadata online.
    """
    strategy_layout_handle = StrategyLayout.get_instance()
    if strategy_layout_handle is None:
        raise ValueError("Strategy layout handle is none in parallel_strategy_checkpoint!!!")
    strategy_layout_handle.enable_save_strategy_online()


def clear_strategy_metadata():
    """Clear all saved strategy metadata on the C++ side."""
    strategy_layout_handle = StrategyLayout.get_instance()
    if strategy_layout_handle is None:
        raise ValueError("Strategy layout handle is none in parallel_strategy_checkpoint!!!")
    return strategy_layout_handle.clear_strategy_metadata()


class _NetStrategyInfo:
    """
    Describe the strategy information of a network.
    """

    def __init__(self, network, global_layout=None, local_layout=None):
        self._network = network
        self._compile_phase = network.compile_phase
        if global_layout is None or local_layout is None:
            layout_handle = self._get_layout_handle()
            global_layout = layout_handle.global_network_layout()
            local_layout = layout_handle.local_network_layout()
        self._raw_global_layout = global_layout
        self._raw_local_layout = local_layout

    @staticmethod
    def _get_layout_handle():
        """Get strategy handle"""
        layout_handle = StrategyLayout.get_instance()
        if layout_handle is None:
            raise ValueError("Strategy layout handle is none in parallel_strategy_checkpoint!!!")
        return layout_handle

    def get_rank_layout(self, rank_id=None):
        """Get params of the network, global rank or special rank, interface."""
        raw_global_layout = self._get_valid_layout(self._compile_phase, self._raw_global_layout)
        if raw_global_layout is None:
            return None
        global_layout = self._extract_layout_metadata(raw_global_layout)
        if rank_id is not None:
            cur_rank_layout = {rank_id: global_layout[rank_id]}
            self._layout_to_string(cur_rank_layout)
            return cur_rank_layout
        self._layout_to_string(global_layout)
        return global_layout

    def get_local_rank_layout(self):
        """Get local rank params of the network, {param_name: param_info[layout]}."""
        raw_local_layout = self._get_valid_layout(self._compile_phase, self._raw_local_layout)
        if raw_local_layout is None:
            return None
        local_layout = self._extract_layout_metadata(raw_local_layout)
        self._layout_to_string(local_layout)
        return local_layout

    @staticmethod
    def _get_valid_layout(phase, layout_dict):
        """Helper: Validate and extract layout by phase."""
        if not phase:
            return None
        layout = layout_dict.get(phase)
        if not layout or all(not v for v in layout.values()):
            return None
        return layout

    def _extract_layout_metadata(self, layout: Dict[int, Dict[str, StrategyInfo]]) -> Dict:
        """Return new layout of special network."""
        new_layout = {}
        for rank_id, param_dict in layout.items():
            new_param_info = {}
            for param_name, param_info in param_dict.items():
                new_param_layout = self._layout_process(param_info)
                new_param_info[param_name] = new_param_layout
            new_layout[rank_id] = new_param_info
        return new_layout

    def _layout_process(self, stra_layout):
        """
        Return the layout list, stra_layout is one of params_info of cur_rank.
        """
        new_dev_mat, counter, new_tensor_map, full_opt_shard = self._get_dev_mat_for_opt_shard(
            stra_layout.opt_weight_shard_size, stra_layout.dev_matrix, stra_layout.tensor_map)
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        alias_name = [alphabet[i] for i in range(len(new_dev_mat))]
        if stra_layout.opt_weight_shard_size == 0:
            new_tensor_map = tuple(tuple(alias_name[len(alias_name) - idx - 1] if idx != -1 else "None" for idx in sub)
                                   for sub in new_tensor_map)
        else:
            info = SimpleNamespace(
                new_dev_mat=new_dev_mat,
                new_tensor_map=new_tensor_map,
                full_opt_shard=full_opt_shard,
                counter=counter,
                alias_name=alias_name
            )
            new_tensor_map = self._get_tensor_map_for_opt_shard(info)
        new_tensor_map = self._compact_tensor_map(new_tensor_map)
        new_dev_mat = tuple(new_dev_mat)
        alias_name = tuple(alias_name)
        layout = Layout(new_dev_mat, alias_name, stra_layout.rank_list)
        final_layout = layout(*new_tensor_map)
        logger.debug("The final layout is %s", final_layout.to_dict())
        cur_param_list = [final_layout, stra_layout.tensor_type, stra_layout.tensor_shape]
        return cur_param_list

    def _get_dev_mat_for_opt_shard(self, opt_shard, dev_mat, tensor_map):
        """generate device matrix for opt shard scenario"""
        if opt_shard == 0:
            return dev_mat, -1, tensor_map, True
        used_dev_num = self._calc_used_dev_num(dev_mat, tensor_map)
        total_dev_num = int(np.prod(np.array(dev_mat)))
        if opt_shard == -1 or used_dev_num * opt_shard == total_dev_num:
            return dev_mat, -1, tensor_map, True
        remain_dev_num = total_dev_num // (used_dev_num * opt_shard)
        used_dev_mat_mask = self._get_used_dev_mat(dev_mat, tensor_map)
        info = SimpleNamespace(
            dev_mat=dev_mat,
            tensor_map=tensor_map,
            counter=-1,
            real_remain_dev_num=1,
            remain_dev_num=remain_dev_num
        )
        for axis, value in enumerate(dev_mat):
            if used_dev_mat_mask[axis]:
                continue
            info.counter = axis
            if info.real_remain_dev_num == info.remain_dev_num:
                return dev_mat, axis, tensor_map, False
            if info.real_remain_dev_num < info.remain_dev_num:
                info.real_remain_dev_num *= value
                continue
            # info.real_remain_dev_num > info.remain_dev_num，split axis.
            return self._split_dev_dim(info)
        if info.real_remain_dev_num == info.remain_dev_num:
            return dev_mat, info.counter, tensor_map, False
        return self._split_dev_dim(info)

    def _get_tensor_map_for_opt_shard(self, info: SimpleNamespace):
        """generate tensor map for opt shard scenario"""

        def idx_to_alias(idx):
            return "None" if idx == -1 else info.alias_name[len(info.alias_name) - idx - 1]

        def entry_to_alias(entry):
            if isinstance(entry, (list, tuple)):
                return tuple(idx_to_alias(i) for i in entry)
            return idx_to_alias(entry)

        used_dev_mat = self._get_used_dev_mat(info.new_dev_mat, info.new_tensor_map)
        if info.full_opt_shard:
            unused_idx = [len(used_dev_mat) - i - 1 for i, used in enumerate(used_dev_mat) if not used]
        else:
            unused_idx = [len(used_dev_mat) - i - 1 for i, used in enumerate(used_dev_mat) if
                          not used and i > info.counter]
        first_entry = info.new_tensor_map[0]
        first_list = list(first_entry) if isinstance(first_entry, (list, tuple)) else [first_entry]
        new_first_list = [dim for dim in first_list + unused_idx if dim != -1]
        first_alias_list = [idx_to_alias(i) for i in new_first_list] or ["None"]
        first_alias = first_alias_list[0] if len(first_alias_list) == 1 else tuple(first_alias_list)
        rest_alias = [entry_to_alias(entry) for entry in info.new_tensor_map[1:]]
        new_tensor_map = tuple([first_alias] + rest_alias)
        return new_tensor_map

    @staticmethod
    def _split_dev_dim(info: SimpleNamespace):
        """Split the counter dimension of dev_mat and adjust tensor_map."""
        dev_mat = info.dev_mat
        counter = info.counter
        splitted_dev_value = dev_mat[counter]
        new_dev_mat_value_first = info.remain_dev_num // (info.real_remain_dev_num // splitted_dev_value)
        new_dev_mat_value_second = splitted_dev_value // new_dev_mat_value_first
        new_dev_mat = dev_mat[:counter] + [new_dev_mat_value_first, new_dev_mat_value_second] + dev_mat[counter + 1:]
        flag = len(new_dev_mat) - 1 - counter
        new_tensor_map = [[v if v < flag or v == -1 else v + 1 for v in sub] for sub in info.tensor_map]
        return new_dev_mat, counter, new_tensor_map, False

    @staticmethod
    def _calc_used_dev_num(dev_mat, tensor_map):
        """Count the total number of device nums that have been used."""
        idx_flat = [idx for idx in chain.from_iterable(tensor_map) if idx != -1]
        if not idx_flat:
            return 1
        prod_list = [dev_mat[len(dev_mat) - idx - 1] for idx in idx_flat]
        return int(np.prod(prod_list))

    @staticmethod
    def _get_used_dev_mat(dev_mat, tensor_map) -> List[bool]:
        """List that records whether the device ID is being used or not."""
        used = set()
        for elem in tensor_map:
            if isinstance(elem, (list, tuple)):
                used.update(i for i in elem if i != -1)
            elif elem != -1:
                used.add(elem)
        return [(len(dev_mat) - i - 1) in used for i in range(len(dev_mat))]

    @staticmethod
    def _compact_tensor_map(alias_map: Sequence[StrOrTuple]) -> Tuple[StrOrTuple, ...]:
        """Extend tensor map of 'None'."""

        def _compress(elem: StrOrTuple) -> StrOrTuple:
            if isinstance(elem, (list, tuple)):
                compressed = tuple(_compress(e) for e in elem)
                if len(compressed) == 1:
                    return compressed[0]
                if all(x == 'None' for x in compressed):
                    return 'None'
                return compressed
            return elem

        return tuple(_compress(e) for e in alias_map)

    @staticmethod
    def _layout_to_string(layout_info):
        """Print layout info."""
        for rank_id, param_layout in layout_info.items():
            logger.info("rank_id=%s", rank_id)
            for param_name, cur_param_list in param_layout.items():
                final_layout, param_type, global_shape = cur_param_list
                logger.info("param_name=%s: [param_layout=%s, param_type=%s, global_shape=%s]",
                            param_name, final_layout.to_dict(), param_type, global_shape)
            logger.info("\n")
