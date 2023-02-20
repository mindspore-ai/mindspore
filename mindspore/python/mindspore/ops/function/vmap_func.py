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
"""Defines vmap function."""
from mindspore.ops.composite import _Vmap
from mindspore._c_expression import VmapGeneralPreprocess_, VmapGeneralRulePyAdapter_

__all__ = ['vmap']
vmap_instance = _Vmap()


def vmap(fn, in_axes=0, out_axes=0):
    r"""
    Vectorizing map (vmap) is a kind of higher-order function to map `fn` along the parameter axes.

    Vmap is pioneered by Jax and it removes the restriction of batch dimension on the operator, and provides a
    more convenient and unified operator expression. Moreover, it allows users to composite with other functional
    modules such as :func:`mindspore.grad`, to improve the development efficiency. In addition, the vectorizing
    map does not execute loops outside the function, but sinks loops into the primitive operations of the function
    for better performance. When combined with `Graph Kernel Fusion`, operational efficiency would be further improved.

    .. warning::
        This is an experimental prototype that is subject to change and/or delete.

    Note:
        1. The power of vmap comes from the implementation of VmapRules of primitives. Although we have designed a
        generalized rule for user custom operators, we can not guarantee that it works well for all operators,
        please be aware the risk of use. If you want to achieve a better performance, please refer to the tutorial to
        implement the specific VmapRule for the custom operator, which won't take too much time.
        2. When calling the random number generation methods within the scope of vmap, the same random number is
        generated among vector functions each time. If you expect each vector branch to use different random numbers,
        you need to generate batch random numbers externally in advance and then transfer them to vmap.

    Args:
        fn (Union[Cell, Function, CellList]): Function to be mapped along the parameter axes, which takes at least one
            argument and returns one or more Tensors or the type of data supported by the MindSpore Tensor. When it is
            a CellList, the model ensembling scenario, please make sure that the structure of each cell is the same
            and the number of cells is consistent with the sizes of the mapped axes (`axis_size`).
        in_axes (Union[int, list, tuple]): Specifies which dimensions (axes) of the inputs should be mapped over.
            If `in_axes` is an integer, all arguments of `fn` are mapped over according to this axis index. If `in_axes`
            is a tuple or list, which only composed of integers or Nones and the length should equal to the number of
            positional arguments to `fn`, indicates which axis to map for each corresponding positional argument.
            Note that, axis integers must be in range :math:`[-ndim, ndim)` for each argument, where `ndim` is the
            number of dimensions of the corresponding argument.  None means not mapping along any axis. Also the
            mapping axis index of the `in_axes` must have at least one positional parameter not None. The sizes of
            the mapped axes (`axis_size`) for all arguments must be equal. Default: 0.
        out_axes (Union[int, list, tuple]): Specifies where the mapped dimensions (axes) should appear in the
            outputs. If `out_axes` is an integer, all outputs of `fn` are specified according to this axis. If
            `out_axes` is a tuple or list, which only composed of integers or Nones. And its length also should be equal
            to the number of outputs of `fn`. Note that, axis integers must be in range :math:`[-ndim, ndim)` for each
            output, where `ndim` is the dimension of the output of the `vmap`-mapped function. All outputs with a
            non-None mapped axis must specify a non-None `out_axes`, and if outputs with None mapped axis specifies
            a non-None `out_axes`, the result broadcasts across the mapped axis. Default: 0.

    Returns:
        Function, returns the Vectorized/Batched version function of `fn`. The arguments and outputs of this function
        correspond to those of `fn`, but it adds an extra batch dimension at positions specified by `in_axes` and
        `out_axes`.

    Raises:
        RuntimeError: If base elements in `in_axes` or `out_axes` are not a None or an integer.
            If the all base elements in `in_axes` or `out_axes` are None.
            If `in_axes` is not single integer, and the length of `in_axes` is not equal to the arguments sizes.
            If `out_axes` is not single integer, and the length of `out_axes` is not equal to the outputs sizes.
            If the `axis_size` of each arguments in the scope of `vmap` are not equal.
            If the axis in `in_axes` or `out_axes` is out of bounds.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from mindspore import Tensor
        >>> from mindspore import vmap
        >>> def test_vmap(x, y, z):                                              # ([a],[a],[a]) -> [a]
        ...     return x + y + z
        >>> x = Tensor(np.array([[1, 2], [3, 4], [5, 6]]).astype(np.float32))    # [b, a]
        >>> y = Tensor(np.array([[-3, -2, -1], [3, 2, 1]]).astype(np.float32))   # [a, b]
        >>> z = Tensor(np.array([0, 3]).astype(np.float32))                      # [a]
        >>> output = vmap(test_vmap, in_axes=(0, 1, None), out_axes=1)(x, y, z)  # ([b, a],[a, b],[a]) -> [a, b]
        >>> print(output)
        [[-2  1  4]
         [ 8  9 10]]
    """
    return vmap_instance(fn, in_axes, out_axes)


class _VmapGeneralPreprocess(VmapGeneralPreprocess_):
    """
    General preprocessing of VmapRules. If the source axes of all inputs are `None`,
    means that vectorization is not performed, taking out the original input and call
    the primitive directly.
    """
    def __init__(self):
        VmapGeneralPreprocess_.__init__(self, "VmapGeneralPreprocess")


class _VmapGeneralRule(VmapGeneralRulePyAdapter_):
    """
    General rule python adapter is a adapter for general rule in c++. Some operators can
    implement loop-stack method in their vmaprule by calling this adapter.
    """
    def __init__(self, prim, axis_size):
        VmapGeneralRulePyAdapter_.__init__(self, 'vmapgeneralrule', prim, axis_size)
