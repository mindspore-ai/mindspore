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
"""Auto mixed precision."""
from __future__ import absolute_import

import mindspore as ms
from mindspore import nn
from mindspore import _checkparam as validator
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.cell_wrapper import _TrainGradAccuStepCell
from mindspore.nn.wrap.loss_scale import _TrainGradAccuWithLossScaleCell
from mindspore.ops import functional as F
from mindspore.parallel._utils import _get_pipeline_stages
from mindspore.train.loss_scale_manager import DynamicLossScaleManager, LossScaleManager
from mindspore import boost, context
from mindspore.ops import operations as P
from mindspore.ops import Primitive
from mindspore import log as logger


AMP_WHITE_LIST = [
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.Conv1dTranspose,
    nn.Conv2dTranspose,
    nn.Conv3dTranspose,
    nn.Dense,
    nn.LSTMCell,
    nn.RNNCell,
    nn.GRUCell,
    P.Conv2D,
    P.Conv3D,
    P.Conv2DTranspose,
    P.Conv3DTranspose,
    P.Conv2DBackpropInput,
    P.MatMul,
    P.BatchMatMul,
    P.PReLU,
    P.ReLU,
    P.Ger
]


AMP_BLACK_LIST = [
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.LayerNorm
]

MS_AMP_BY_REWRITE = False
_amp_cast_op = P.Cast


class _OutputTo16(nn.Cell):
    """Wrap cell for amp. Cast network output back to float16."""
    def __init__(self, backbone, dtype=mstype.float16):
        super(_OutputTo16, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self.dtype = dtype
        self._get_attr_from_cell(backbone)

    def construct(self, *args, **kwargs):
        return F.cast(self._backbone(*args, **kwargs), self.dtype)


class _OutputTo32(nn.Cell):
    """Wrap loss for amp. Cast network output back to float32."""
    def __init__(self, backbone):
        super(_OutputTo32, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._get_attr_from_cell(backbone)

    def construct(self, *args, **kwargs):
        out = self._backbone(*args, **kwargs)
        return F.mixed_precision_cast(mstype.float32, out)



def _allow_mix_precision(node, allowed_list, dtype) -> bool:
    """
    Check whether current node need do mix precision. Follow conditions need to be satisfied:
        1) Type of node is one of (Primitive, nn.Cell)
        2) Node is not Cast Op
        3) to_float(mindspore.float16) is not set in Cell
    """
    node_inst = node.get_instance()
    if node_inst in allowed_list:
        return True
    if node.get_targets() is None:
        return False
    if not issubclass(node.get_instance_type(), (Primitive, nn.Cell)):
        return False
    if isinstance(node_inst, _amp_cast_op):
        return False
    if issubclass(node.get_instance_type(), nn.Cell):
        # if cell is already in allowed_list, it means to_float() is set by amp.
        # if cell is not in allowed_list, but has to_float(),
        # it means to_float() is set by user.
        to_float_flag = "bf16" if dtype == mstype.bfloat16 else "fp16"
        if hasattr(node_inst, to_float_flag) and getattr(node_inst, to_float_flag):
            return False
    allowed_list.append(node.get_instance())
    return True


def _insert_cast_operator_process(node, dtype):
    """insert cast for operators in white_list."""
    dtype_str = "mindspore.bfloat16" if dtype == mstype.bfloat16 else "mindspore.float16"
    new_cast_node = None
    stree = node.get_symbol_tree()
    # insert cast fp16/bf16 before the primitive operators
    if issubclass(node.get_instance_type(), Primitive):
        for idx, arg in enumerate(node.get_args()):
            position = stree.before(node)
            new_node = _amp_cast_op()
            cast_args = ms.rewrite.ScopedValue.create_name_values([arg.value, dtype_str], [arg.scope, ""])
            arg_provider = node.get_handler().get_arg_providers()[idx]
            if arg_provider and len(arg_provider[0].get_target_users(arg_provider[1])) > 1:
                cast_targets = [stree.unique_name(str(arg))]
            else:
                cast_targets = ms.rewrite.ScopedValue.create_name_values([arg.value], [arg.scope])
            new_cast_node = ms.rewrite.Node.create_call_cell(new_node,
                                                             targets=cast_targets,
                                                             args=cast_args,
                                                             name='incast_{}{}'.format(node.get_name(), idx))
            stree.insert(position, new_cast_node)
            node.set_arg_by_node(idx, new_cast_node)
    # insert cast fp16/bf16 before the Cell operators
    elif issubclass(node.get_instance_type(), nn.Cell):
        node.get_instance().to_float(dtype)
    # ignore if subclass is not one of (Primitive, nn.Cell)
    else:
        return

    # insert cast float32 after the operators
    position = stree.after(node)
    new_node = _amp_cast_op()
    cast_args = ms.rewrite.ScopedValue.create_name_values([node.get_targets()[0].value,
                                                           "mindspore.float32"])
    new_cast_node = ms.rewrite.Node.create_call_cell(new_node,
                                                     targets=[node.get_targets()[0]],
                                                     args=cast_args,
                                                     name='outcast_{}'.format(node.get_name()))
    # insert node & unique names
    stree.insert(position, new_cast_node)
    # update argument names
    for user in node.get_users():
        if user.get_name() == new_cast_node.get_name():
            continue
        for idx, arg in enumerate(user.get_args()):
            if arg == node.get_targets()[0]:
                user.set_arg_by_node(idx, new_cast_node)


def _insert_cast_operator_white_list(stree, white_list, dtype):
    """insert cast for operators in white_list."""
    allowed_list = []
    # Ignore if net called ".to_float(dtype)"
    net = stree.get_handler().get_origin_network()
    to_float_flag = "bf16" if dtype == mstype.bfloat16 else "fp16"
    if isinstance(net, nn.Cell) and hasattr(net, to_float_flag) and getattr(net, to_float_flag):
        return
    node_list = []
    node_list.extend(list(stree.nodes()))
    while node_list:
        node = node_list.pop()
        if node.get_node_type() == ms.rewrite.NodeType.CellContainer:
            if MS_AMP_BY_REWRITE:
                _insert_cast_for_cell_container(node, dtype, allowed_list, white_list=white_list)
            for n in node.get_handler().node_list:
                if n.get_node_type() == ms.rewrite.NodeType.Tree:
                    _insert_cast_operator_white_list(ms.rewrite.TreeNodeHelper.get_sub_tree(ms.rewrite.Node(n)),
                                                     white_list, dtype)
        elif node.get_node_type() == ms.rewrite.NodeType.Tree:
            substree = ms.rewrite.TreeNodeHelper.get_sub_tree(node)
            _insert_cast_operator_white_list(substree, white_list, dtype)
        elif node.get_node_type() in [ms.rewrite.NodeType.CallFunction, ms.rewrite.NodeType.ControlFlow]:
            if isinstance(node.get_handler(), ms.rewrite.node.NodeManager):
                nodes = [ms.rewrite.Node(n) for n in node.get_handler().nodes()]
                node_list.extend(nodes)
        elif node.get_instance_type() in white_list and _allow_mix_precision(node, allowed_list, dtype):
            _insert_cast_operator_process(node, dtype)


def _insert_cast_for_cell_container(cell_container, dtype, allowed_list, *, white_list=None, black_list=None):
    """
    Insert cast for cell containers.
    Only one of white_list and black_list can be set.
    """

    class CastNet(nn.Cell):
        """Cast net"""
        def __init__(self, dtype):
            super().__init__()
            self.cast = _amp_cast_op()
            self.dtype = dtype

        def construct(self, x):
            return self.cast(x, self.dtype)

    cast_flag = False
    current_node = None
    stree = cell_container.get_symbol_tree()
    for node in cell_container.get_handler().nodes():
        current_node = ms.rewrite.Node(node)
        if (white_list is not None and current_node.get_instance_type() in white_list) or \
           (black_list is not None and current_node.get_instance_type() not in black_list) and \
           (_allow_mix_precision(current_node, allowed_list, dtype)):
            cast_flag = True
            current_node.get_instance().to_float(dtype)
        elif cast_flag:
            # cast next node back to float32
            current_node.get_instance().to_float(mstype.float32)
            cast_flag = False
    if cast_flag and current_node:
        # if last node in cell_container is casted to fp16/bf16, insert a cast node to cast value back to fp32
        cast_node = ms.rewrite.Node.create_call_cell(cell=CastNet(mstype.float32),
                                                     args=[current_node.get_targets()[0]],
                                                     targets=[current_node.get_targets()[0]],
                                                     name=f"outcast_{cell_container.get_name()}")
        stree.insert(stree.after(current_node), cast_node)


def _need_removed_cast_pair(node, dtype):
    """check whether the cast pairs should be removed."""
    dtype_str = "mindspore.bfloat16" if dtype == mstype.bfloat16 else "mindspore.float16"
    cast_dtypes = ms.rewrite.ScopedValue.create_name_values([dtype_str, "mindspore.float32"])
    cast_dtype_f16 = cast_dtypes[0]
    cast_dtype_f32 = cast_dtypes[1]
    # current node should be Cast Op to float32
    if node.get_instance_type() != _amp_cast_op:
        return False
    node_cast_type = node.get_args()[1]
    if node_cast_type != cast_dtype_f32:
        return False
    # all user nodes should be Cast Op to dtype or Cell with to_float(dtype)
    if not node.get_users():
        return False
    all_nodes = [ms.rewrite.Node(n) for n in node.get_handler().get_node_manager().nodes()]
    for user in node.get_users():
        # If ControlFlow node(if statement) exists between current node and user node,
        # cast pair should not be removed.
        middle_nodes = all_nodes[all_nodes.index(node): all_nodes.index(user)]
        if any([n.get_node_type() == ms.rewrite.NodeType.ControlFlow for n in middle_nodes]):
            return False
        if isinstance(user.get_instance(), nn.Cell):
            to_float_flag = "bf16" if dtype == mstype.bfloat16 else "fp16"
            if not (hasattr(user.get_instance(), to_float_flag) and getattr(user.get_instance(), to_float_flag)):
                return False
        elif user.get_instance_type() == _amp_cast_op:
            user_cast_type = user.get_args()[1]
            if user_cast_type != cast_dtype_f16:
                return False
        else:
            return False
    return True


def _removed_cast_pair_process(cast_f32_node):
    """remove the duplicated cast operators."""
    stree = cast_f32_node.get_symbol_tree()
    cast_f32_users = cast_f32_node.get_users()
    # remove cast f16 nodes
    for user_node in cast_f32_users:
        if user_node.get_instance_type() == _amp_cast_op:
            cast_f16_node = user_node
            # modify arguments using cast_f16's target[0] to cast_f32's args[0], which is f16 type
            for cast_f16_user in cast_f16_node.get_users():
                for idx, arg in enumerate(cast_f16_user.get_args()):
                    if arg == cast_f16_node.get_targets()[0]:
                        cast_f16_user.set_arg(idx, cast_f32_node.get_args()[0])
            stree.erase(cast_f16_node)
        # update args of cell f16 nodes
        elif isinstance(user_node.get_instance(), nn.Cell):
            cell_f16_node = user_node
            for idx, arg in enumerate(cell_f16_node.get_args()):
                if arg == cast_f32_node.get_targets()[0]:
                    cell_f16_node.set_arg(idx, cast_f32_node.get_args()[0])
    # remove the cast f32 node
    stree.erase(cast_f32_node)


def _remove_duplicated_cast(stree, dtype):
    """remove the duplicated cast operators."""
    node_list = []
    node_list.extend(list(stree.nodes()))
    while node_list:
        node = node_list.pop()
        if node.get_node_type() == ms.rewrite.NodeType.CellContainer:
            for n in node.get_handler().node_list:
                if n.get_node_type() == ms.rewrite.NodeType.Tree:
                    _remove_duplicated_cast(ms.rewrite.TreeNodeHelper.get_sub_tree(ms.rewrite.Node(n)), dtype)
        elif node.get_node_type() == ms.rewrite.NodeType.Tree:
            substree = ms.rewrite.TreeNodeHelper.get_sub_tree(node)
            _remove_duplicated_cast(substree, dtype)
        elif node.get_node_type() in [ms.rewrite.NodeType.CallFunction, ms.rewrite.NodeType.ControlFlow]:
            if isinstance(node.get_handler(), ms.rewrite.node.NodeManager):
                nodes = [ms.rewrite.Node(n) for n in node.get_handler().nodes()]
                node_list.extend(nodes)
        elif _need_removed_cast_pair(node, dtype):
            _removed_cast_pair_process(node)


def _auto_white_list(network, white_list, dtype):
    """process the white list of network."""
    stree = ms.rewrite.SymbolTree.create(network)
    _insert_cast_operator_white_list(stree, white_list, dtype)
    _remove_duplicated_cast(stree, dtype)
    return stree.get_network()


def _insert_cast_operator_black_list(stree, black_list, dtype):
    """insert cast for operators not in black_list."""
    allowed_list = []
    # Ignore if net called ".to_float(dtype)"
    net = stree.get_handler().get_origin_network()
    to_float_flag = "bf16" if dtype == mstype.bfloat16 else "fp16"
    if isinstance(net, nn.Cell) and hasattr(net, to_float_flag) and getattr(net, to_float_flag):
        return
    for node in stree.nodes(all_nodes=True):
        if node.get_targets() is None:
            continue
        if node.get_node_type() == ms.rewrite.NodeType.CellContainer:
            _insert_cast_for_cell_container(node, dtype, allowed_list, black_list=black_list)
        elif isinstance(node.get_handler().get_node_manager(), ms.rewrite.node.CellContainer):
            # nodes in CellContainer are processed by _insert_cast_for_cell_container
            continue
        elif node.get_instance_type() not in black_list and _allow_mix_precision(node, allowed_list, dtype):
            _insert_cast_operator_process(node, dtype)


def _remove_duplicated_cast_rewrite(stree, dtype):
    """remove the duplicated cast operators."""
    for node in stree.nodes(all_nodes=True):
        if _need_removed_cast_pair(node, dtype):
            user_nodes = node.get_users()
            # remove cast f16 nodes
            for user_node in user_nodes:
                if user_node.get_instance_type() == _amp_cast_op:
                    stree.erase(user_node)
            # remove the cast f32 node
            stree.erase(node)


def _auto_black_list_rewrite(network, black_list, dtype):
    stree = ms.rewrite.SymbolTree.create(network)
    _insert_cast_operator_black_list(stree, black_list, dtype)
    _remove_duplicated_cast_rewrite(stree, dtype)
    return stree.get_network()


def _auto_black_list(network, black_list, dtype):
    """process the black list of network."""
    network.to_float(dtype)
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        if isinstance(subcell, tuple(black_list)):
            network._cells[name] = _OutputTo16(subcell.to_float(mstype.float32), dtype)
            change = True
        else:
            _auto_black_list(subcell, black_list, dtype)
    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())
    return network


def auto_mixed_precision(network, amp_level="O0", dtype=mstype.float16):
    """
    Returns a network processed with auto mixed precision.

    This interface will automatically perform mixed-precision processing on the input network, and the cells
    and operators in the processed network will add precision conversion operations to calculate with lower
    precision: ``mstype.float16`` or ``mstype.bfloat16`` . Inputs and parameters of cells and operators are
    converted to lower precision float, and calculation results are converted back to full precision float,
    i.e. ``mstype.float32`` .

    The framework has a set of built-in blacklists and whitelists, and the `amp_level` determines which cells and
    operators are specifically converted.

    The current built-in whitelist contents are:

    [:class:`mindspore.nn.Conv1d`, :class:`mindspore.nn.Conv2d`, :class:`mindspore.nn.Conv3d`,
    :class:`mindspore.nn.Conv1dTranspose`, :class:`mindspore.nn.Conv2dTranspose`,
    :class:`mindspore.nn.Conv3dTranspose`, :class:`mindspore.nn.Dense`, :class:`mindspore.nn.LSTMCell`,
    :class:`mindspore.nn.RNNCell`, :class:`mindspore.nn.GRUCell`, :class:`mindspore.ops.Conv2D`,
    :class:`mindspore.ops.Conv3D`, :class:`mindspore.ops.Conv2DTranspose`,
    :class:`mindspore.ops.Conv3DTranspose`, :class:`mindspore.ops.MatMul`, :class:`mindspore.ops.BatchMatMul`,
    :class:`mindspore.ops.PReLU`, :class:`mindspore.ops.ReLU`, :class:`mindspore.ops.Ger`]

    The current built-in blacklist contents are:

    [:class:`mindspore.nn.BatchNorm1d`, :class:`mindspore.nn.BatchNorm2d`, :class:`mindspore.nn.BatchNorm3d`,
    :class:`mindspore.nn.LayerNorm`]

    For details on automatic mixed precision, refer to
    `Automatic Mix Precision <https://www.mindspore.cn/tutorials/en/master/advanced/mixed_precision.html>`_ .

    Note:
        - Repeatedly calling mixed-precision interfaces, such as `custom_mixed_precision` and `auto_mixed_precision`,
          can result in a larger network hierarchy and slower performance.
        - If interfaces like `Model` and `build_train_network` is used to train the network which is converted by
          mixed-precision interfaces such as `custom_mixed_precision` and `auto_mixed_precision`, `amp_level`
          need to be configured to ``O0`` to avoid the duplicated accuracy conversion.

    Args:
        network (Cell): Definition of the network.
        amp_level (str): Supports ["O0", "O1", "O2", "O3"]. Default: ``"O0"`` .

            - "O0": Do not change.
            - "O1": Convert cells and operators in whitelist to lower precision operations, and keep full
              precision operations for the rest.
            - "O2": Keep full precision operations for cells and operators in blacklist, and convert the rest
              to lower precision operations.
            - "O3": Cast network to lower precision.

        dtype (Type): The type used in lower precision calculations, can be ``mstype.float16`` or ``mstype.bfloat16`` ,
            default: ``mstype.float16`` .

    Raises:
        TypeError: If `network` is not a Cell.
        ValueError: If `dtype` is not one of ``mstype.float16`` , ``mstype.bfloat16`` .
        ValueError: If `amp_level` is not within the supported range.

    Examples:
        >>> from mindspore import amp
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> network = LeNet5()
        >>> amp_level = "O1"
        >>> net = amp.auto_mixed_precision(network, amp_level)
    """
    if not isinstance(network, nn.Cell):
        raise TypeError("The network type should be Cell.")

    if dtype not in (mstype.float16, mstype.bfloat16):
        raise ValueError(f"The dtype should be one of (mstype.float16, mstype.bfloat16), but got {dtype}.")

    if amp_level == "O0":
        return network

    # Return network if the same amp level has already been configurated
    if getattr(network, "_amp_level") in ("O1", "O2", "O3"):
        logger.warning(f"The network's auto mixed-precision level is adjusted from {getattr(network, '_amp_level')} "
                       f"to {amp_level}, and repeated calls to mixed-precision interfaces can cause performance "
                       f"degradation.")

    if amp_level == "O1":
        network = _auto_white_list(network, AMP_WHITE_LIST, dtype)
    elif amp_level == "O2":
        if MS_AMP_BY_REWRITE:
            network = _auto_black_list_rewrite(network, AMP_BLACK_LIST, dtype)
        else:
            network = _auto_black_list(network, AMP_BLACK_LIST, dtype)
            network = _OutputTo32(network)
    elif amp_level == "O3":
        if MS_AMP_BY_REWRITE:
            network = _auto_black_list_rewrite(network, [], dtype)
        else:
            network.to_float(dtype)
            network = _OutputTo32(network)
    else:
        raise ValueError("The amp level {} is not supported".format(amp_level))

    setattr(network, "_amp_level", amp_level)

    return network


def _do_keep_batchnorm_fp32(network):
    """Do keep batchnorm fp32."""
    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        elif isinstance(subcell, nn.Cell) and isinstance(subcell, tuple(AMP_BLACK_LIST)):
            network._cells[name] = _OutputTo16(subcell.to_float(mstype.float32))
            change = True
        else:
            _do_keep_batchnorm_fp32(subcell)
    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())


_config_level = {
    "O0": {
        "keep_batchnorm_fp32": False,
        "cast_model_type": mstype.float32,
        "loss_scale_manager": None},
    "O1": {
        "keep_batchnorm_fp32": False,
        "cast_model_type": mstype.float32,
        "loss_scale_manager": None},
    "O2": {
        "keep_batchnorm_fp32": True,
        "cast_model_type": mstype.float16,
        "loss_scale_manager": DynamicLossScaleManager()},
    "O3": {
        "keep_batchnorm_fp32": False,
        "cast_model_type": mstype.float16,
        "loss_scale_manager": None}}


def _check_kwargs(key_words):
    """Check kwargs."""
    for arg in key_words:
        if arg not in ['cast_model_type', 'keep_batchnorm_fp32', 'loss_scale_manager']:
            raise ValueError(f"Unsupported arg '{arg}'")

    if 'cast_model_type' in key_words:
        validator.check_type_name('cast_model_type', key_words['cast_model_type'],
                                  [mstype.float16, mstype.float32], None)
    if 'keep_batchnorm_fp32' in key_words:
        validator.check_value_type('keep_batchnorm_fp32', key_words['keep_batchnorm_fp32'], bool)
    if 'loss_scale_manager' in key_words:
        loss_scale_manager = key_words['loss_scale_manager']
        if loss_scale_manager:
            validator.check_value_type('loss_scale_manager', loss_scale_manager,
                                       [LossScaleManager, boost.GroupLossScaleManager])


def _check_level(level, boost_level):
    """Check level."""
    if not isinstance(level, str):
        raise TypeError("The argument `level` must be a string in ['O0', 'O1', 'O2', 'O3', 'auto'], \
                         but got type {}.".format(type(level)))
    validator.check('level', level, "", ['O0', 'O1', 'O2', 'O3', 'auto'], validator.IN)
    validator.check('boost_level', boost_level, "", ['O0', 'O1', 'O2'], validator.IN)

    if level == "auto":
        device_target = context.get_context('device_target')
        if device_target == "GPU":
            level = "O2"
        elif device_target == "Ascend":
            level = "O3"
        else:
            raise ValueError("Level `auto` only support when `device_target` is GPU or Ascend.")

    enable_boost = False
    if boost_level in ["O1", "O2"]:
        enable_boost = True

    return level, enable_boost


def _add_loss_network(network, loss_fn, cast_model_type):
    """Add loss network."""

    class WithLossCell(nn.Cell):
        """Wrap loss for amp. Cast network output back to float32."""
        def __init__(self, backbone, loss_fn):
            super(WithLossCell, self).__init__(auto_prefix=False)
            self._backbone = backbone
            self._loss_fn = loss_fn
            self._get_attr_from_cell(backbone)

        def construct(self, data, label):
            out = self._backbone(data)
            label = F.mixed_precision_cast(mstype.float32, label)
            return self._loss_fn(F.mixed_precision_cast(mstype.float32, out), label)

    validator.check_value_type('loss_fn', loss_fn, nn.Cell)
    if cast_model_type == mstype.float16:
        network = WithLossCell(network, loss_fn)
    else:
        network = nn.WithLossCell(network, loss_fn)
    return network


def _is_grad_accumulation(mcell):
    if mcell.cls_name == "GradAccumulationCell":
        return True
    for cell in mcell.cells():
        if _is_grad_accumulation(cell):
            return True
    return False


def _auto_mixed_precision_process(network, config, level):
    """Auto mixed precision process."""
    if MS_AMP_BY_REWRITE:
        if config["cast_model_type"] == mstype.float16 or level == "O2":
            level = "O2" if config["keep_batchnorm_fp32"] else "O3"
        elif config["cast_model_type"] == mstype.float32 and level in ("O2", "O3"):
            # cast_model_type set by kwargs
            level = "O0"
        network = auto_mixed_precision(network, level)
    else:
        if config["cast_model_type"] == mstype.float16:
            network.to_float(mstype.float16)

            if config["keep_batchnorm_fp32"]:
                _do_keep_batchnorm_fp32(network)
        elif not config["keep_batchnorm_fp32"] and level == "O2":
            network.to_float(mstype.float16)
        elif config["cast_model_type"] == mstype.float32 and level in ("O2", "O3"):
            pass
        else:
            network = auto_mixed_precision(network, level)
    return network


def build_train_network(network, optimizer, loss_fn=None, level='O0', boost_level='O0', **kwargs):
    """
    Build the mixed precision training cell automatically.

    Note:
        - After using `custom_mixed_precision` or `auto_mixed_precision` for precision conversion, it is not supported
          to perform the precision conversion again. If  `build_train_network` is used to train a converted network,
          `level` need to be configured to ``O0`` to avoid the duplicated accuracy conversion.

    Args:
        network (Cell): Definition of the network.
        optimizer (:class:`mindspore.nn.Optimizer`): Define the optimizer to update the Parameter.
        loss_fn (Union[None, Cell]): Define the loss function. If None, the `network` should have the loss inside.
            Default: ``None`` .
        level (str): Supports ['O0', 'O1', 'O2', 'O3', 'auto']. Default: ``'O0'`` .

            - 'O0': Do not change.
            - 'O1': Cast the operators in white_list to float16, the remaining operators are kept in float32.
              The operators in the whitelist: [Conv1d, Conv2d, Conv3d, Conv1dTranspose, Conv2dTranspose,
              Conv3dTranspose, Dense, LSTMCell, RNNCell, GRUCell, MatMul, BatchMatMul, PReLU, ReLU, Ger].
            - 'O2': Cast network to float16, keep batchnorm and `loss_fn` (if set) run in float32,
              using dynamic loss scale.
            - 'O3': Cast network to float16, with additional property `keep_batchnorm_fp32=False` .
            - 'auto': Set to level to recommended level in different devices. Set level to 'O2' on GPU, Set
              level to 'O3' Ascend. The recommended level is chosen by the export experience, not applicable to all
              scenarios. User should specify the level for special network.

            'O2' is recommended on GPU, 'O3' is recommended on Ascend. Property of `keep_batchnorm_fp32`,
            `cast_model_type` and `loss_scale_manager` determined by `level` setting may be overwritten by settings in
            `kwargs`.

        boost_level (str): Option for argument `level` in `mindspore.boost` , level for boost mode
            training. Supports ['O0', 'O1', 'O2']. Default: ``'O0'`` .

            - 'O0': Do not change.
            - 'O1': Enable the boost mode, the performance is improved by about 20%, and
              the accuracy is the same as the original accuracy.
            - 'O2': Enable the boost mode, the performance is improved by about 30%, and
              the accuracy is reduced by less than 3%.

            If 'O1' or 'O2' mode is set, the boost related library will take effect automatically.

        cast_model_type (:class:`mindspore.dtype`): Supports `mstype.float16` or `mstype.float32` . If set, the
            network will be casted to `cast_model_type` ( `mstype.float16` or `mstype.float32` ), but not to be casted
            to the type determined by `level` setting.
        keep_batchnorm_fp32 (bool): Keep Batchnorm run in `float32` when the network is set to cast to `float16` . If
            set, the `level` setting will take no effect on this property.
        loss_scale_manager (Union[None, LossScaleManager]): If not None, must be subclass of
            :class:`mindspore.amp.LossScaleManager` for scaling the loss. If set, the `level` setting will
            take no effect on this property.

    Raises:
        ValueError: If device is CPU, property `loss_scale_manager` is not `None` or `FixedLossScaleManager`
            (with property `drop_overflow_update=False` ).

    Examples:
        >>> from mindspore import amp, nn
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> network = LeNet5()
        >>> net_loss = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")
        >>> net_opt = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
        >>> amp_level="O3"
        >>> net = amp.build_train_network(network, net_opt, net_loss, amp_level)
    """
    validator.check_value_type('optimizer', optimizer, (nn.Optimizer, boost.FreezeOpt,
                                                        nn.AdaSumByGradWrapCell, nn.AdaSumByDeltaWeightWrapCell))

    level, enable_boost = _check_level(level, boost_level)

    _check_kwargs(kwargs)
    config = dict(_config_level.get(level), **kwargs)

    network = _auto_mixed_precision_process(network, config, level)

    if loss_fn:
        network = _add_loss_network(network, loss_fn, config["cast_model_type"])

    loss_scale = None
    if config["loss_scale_manager"] is not None:
        loss_scale_manager = config["loss_scale_manager"]
        loss_scale = loss_scale_manager.get_loss_scale()
        update_cell = loss_scale_manager.get_update_cell()
        if update_cell is not None:
            # only cpu not support `TrainOneStepWithLossScaleCell` for control flow.
            if not context.get_context("enable_ge") and context.get_context("device_target") == "CPU":
                raise ValueError("Only `loss_scale_manager=None` or "
                                 "`loss_scale_manager=FixedLossScaleManager(drop_overflow_update=False)`"
                                 "are supported on device `CPU`. ")
            if _get_pipeline_stages() > 1 or _is_grad_accumulation(network):
                network = _TrainGradAccuWithLossScaleCell(network, optimizer,
                                                          scale_sense=update_cell).set_train()
            elif enable_boost:
                network = boost.BoostTrainOneStepWithLossScaleCell(network, optimizer,
                                                                   scale_sense=update_cell).set_train()
            else:
                network = nn.TrainOneStepWithLossScaleCell(network, optimizer,
                                                           scale_sense=update_cell).set_train()
            return network
    if _get_pipeline_stages() > 1 or _is_grad_accumulation(network):
        network = _TrainGradAccuStepCell(network, optimizer).set_train()
    elif enable_boost:
        network = boost.BoostTrainOneStepCell(network, optimizer, loss_scale).set_train()
    else:
        network = nn.TrainOneStepCell(network, optimizer, loss_scale).set_train()
    return network


def get_white_list():
    """
    Provide a copy of internal white list used by auto mixed precision.

    The current built-in whitelist contents are:

    [:class:`mindspore.nn.Conv1d`, :class:`mindspore.nn.Conv2d`, :class:`mindspore.nn.Conv3d`,
    :class:`mindspore.nn.Conv1dTranspose`, :class:`mindspore.nn.Conv2dTranspose`,
    :class:`mindspore.nn.Conv3dTranspose`, :class:`mindspore.nn.Dense`, :class:`mindspore.nn.LSTMCell`,
    :class:`mindspore.nn.RNNCell`, :class:`mindspore.nn.GRUCell`, :class:`mindspore.ops.Conv2D`,
    :class:`mindspore.ops.Conv3D`, :class:`mindspore.ops.Conv2DTranspose`,
    :class:`mindspore.ops.Conv3DTranspose`, :class:`mindspore.ops.MatMul`, :class:`mindspore.ops.BatchMatMul`,
    :class:`mindspore.ops.PReLU`, :class:`mindspore.ops.ReLU`, :class:`mindspore.ops.Ger`]

    Returns:
        list, A copy of internal white list.

    Examples:
        >>> from mindspore import amp
        >>> white_list = amp.get_white_list()
        >>> print(white_list)
        [<class 'mindspore.nn.layer.conv.Conv1d'>, <class 'mindspore.nn.layer.conv.Conv2d'>,
         <class 'mindspore.nn.layer.conv.Conv3d'>, <class 'mindspore.nn.layer.conv.Conv1dTranspose'>,
         <class 'mindspore.nn.layer.conv.Conv2dTranspose'>, <class 'mindspore.nn.layer.conv.Conv3dTranspose'>,
         <class 'mindspore.nn.layer.basic.Dense'>, <class 'mindspore.nn.layer.rnn_cells.LSTMCell'>,
         <class 'mindspore.nn.layer.rnn_cells.RNNCell'>, <class 'mindspore.nn.layer.rnn_cells.GRUCell'>,
         <class 'mindspore.ops.operations.nn_ops.Conv2D'>, <class 'mindspore.ops.operations.nn_ops.Conv3D'>,
         <class 'mindspore.ops.operations.nn_ops.Conv2DTranspose'>,
         <class 'mindspore.ops.operations.nn_ops.Conv3DTranspose'>,
         <class 'mindspore.ops.operations.nn_ops.Conv2DBackpropInput'>,
         <class 'mindspore.ops.operations.math_ops.MatMul'>, <class 'mindspore.ops.operations.math_ops.BatchMatMul'>,
         <class 'mindspore.ops.operations.nn_ops.PReLU'>, <class 'mindspore.ops.operations.nn_ops.ReLU'>,
         <class 'mindspore.ops.operations.math_ops.Ger'>]
    """
    white_list = AMP_WHITE_LIST.copy()
    return white_list


def get_black_list():
    """
    Provide a copy of internal black list used by auto mixed precision.

    The current built-in blacklist contents are:

    [:class:`mindspore.nn.BatchNorm1d`, :class:`mindspore.nn.BatchNorm2d`, :class:`mindspore.nn.BatchNorm3d`,
    :class:`mindspore.nn.LayerNorm`]

    Returns:
        list, A copy of internal black list.

    Examples:
        >>> from mindspore import amp
        >>> black_list = amp.get_black_list()
        >>> print(black_list)
        [<class 'mindspore.nn.layer.normalization.BatchNorm1d'>, <class 'mindspore.nn.layer.normalization.BatchNorm2d'>,
         <class 'mindspore.nn.layer.normalization.BatchNorm3d'>, <class 'mindspore.nn.layer.normalization.LayerNorm'>]
    """
    black_list = AMP_BLACK_LIST.copy()
    return black_list


def custom_mixed_precision(network, *, white_list=None, black_list=None, dtype=mstype.float16):
    """
    Custom mixed precision by setting whitelist or blacklist.
    When the `white_list` is provided, primitives and cells in `white_list` will perform the precision conversion.
    When the `black_list` is provided, cells that are not in `black_list` will perform the pereision conversion.
    Only one of `white_list` and `black_list` should be provided.

    Note:
        - Repeatedly calling mixed-precision interfaces, such as `custom_mixed_precision` and `auto_mixed_precision`,
          can result in a larger network hierarchy and slower performance.
        - If interfaces like `Model` and `build_train_network` is used to train the network which is converted by
          mixed-precision interfaces such as `custom_mixed_precision` and `auto_mixed_precision`, `amp_level`
          need to be configured to ``O0`` to avoid the duplicated accuracy conversion.
        - Primitives for blacklist is not support yet.

    Args:
        network (Cell): Definition of the network.
        white_list (list[Primitive, Cell], optional): White list of custom mixed precision. Defaults: ``None`` , means
            white list is not used.
        black_list (list[Cell], optional): Black list of custom mixed precision. Defaults: ``None`` , means
            black list is not used.
        dtype (Type): The type used in lower precision calculations, can be ``mstype.float16`` or ``mstype.bfloat16`` ,
            default: ``mstype.float16`` .

    Returns:
        network (Cell), A network supporting mixed precision.

    Raises:
        TypeError: The network type is not Cell.
        ValueError: Neither `white_list` nor `black_list` is provided.
        ValueError: If `dtype` is not one of ``mstype.float16`` , ``mstype.bfloat16`` .
        ValueError: Both `white_list` and `black_list` are provided.

    Examples:
        >>> from mindspore import amp, nn
        >>> # Define the network structure of LeNet5. Refer to
        >>> # https://gitee.com/mindspore/docs/blob/master/docs/mindspore/code/lenet.py
        >>> net = LeNet5()
        >>> custom_white_list = amp.get_white_list()
        >>> custom_white_list.append(nn.Flatten)
        >>> net = amp.custom_mixed_precision(net, white_list=custom_white_list)
    """
    if not isinstance(network, nn.Cell):
        raise TypeError("The network type should be Cell.")

    if white_list is None and black_list is None:
        raise ValueError("For custom_mixed_precision, one of white_list and black_list must be provided.")

    if white_list is not None and black_list is not None:
        raise ValueError("For custom_mixed_precision, the white_list or black_list cannot be provided "
                         "at the same time, please provide one or the other.")

    if dtype not in (mstype.float16, mstype.bfloat16):
        raise ValueError(f"The dtype should be one of (mstype.float16, mstype.bfloat16), but got {dtype}.")

    if white_list is not None:
        _list_check(white_list, "white_list")
        network = _auto_white_list(network, white_list, dtype)
    else:
        _list_check(black_list, "black_list")
        if MS_AMP_BY_REWRITE:
            network = _auto_black_list_rewrite(network, black_list, dtype)
        else:
            network = _auto_black_list(network, black_list, dtype)
            network = _OutputTo32(network)
    return network


def _list_check(custom_list: list, list_name: str):
    """
    check whether custom list is valid

    Raises:
        TypeError: The type of custom_list is not list.
        TypeError: The element in custom_list is not a class.
        TypeError: The subclass of element in custom_list is not one of ['Cell', 'Primitive'].
    """
    if not isinstance(custom_list, list):
        raise TypeError(f"The type of {list_name} should be list, but got {type(custom_list)}")

    for elem in custom_list:
        if not isinstance(elem, type):
            raise TypeError(f"The element in {list_name} should be a class, but got {elem}")

        if list_name == "white_list" and not issubclass(elem, nn.Cell) and not issubclass(elem, Primitive):
            raise TypeError(f"The subclass of element in {list_name} should be one of 'Cell' and 'Primitive', "
                            f"but got {elem}")

        if list_name == "black_list" and not issubclass(elem, nn.Cell):
            raise TypeError(f"The subclass of element in {list_name} should be one of 'Cell', but got {elem}")

    if list_name == 'black_list':
        for elem in AMP_BLACK_LIST:
            if elem not in custom_list:
                logger.warning(f"{elem} is removed from internal black list.")

def _config_amp(*, enable_rewrite: bool = None, cast_op: type = None): # pylint: disable=unused-variable
    """Configure auto mixed precision."""
    global MS_AMP_BY_REWRITE
    global _amp_cast_op

    if enable_rewrite is not None:
        MS_AMP_BY_REWRITE = enable_rewrite

    if cast_op is not None:
        _amp_cast_op = cast_op
