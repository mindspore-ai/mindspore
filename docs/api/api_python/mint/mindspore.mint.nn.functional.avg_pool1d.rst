mindspore.mint.nn.functional.avg_pool1d
=======================================

.. py:function:: mindspore.mint.nn.functional.avg_pool1d(input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True)

    在输入Tensor上应用1D平均池化，输入Tensor可以看作是由一系列1D平面组成的。

    一般地，输入的shape为 :math:`(N_{in}, C_{in}, L_{in})` ，输出 :math:`(L_{in})` 维度的区域平均值。给定 `kernel_size` 为 :math:`ks = l_{ker}` 和 `stride` 为 :math:`s = s_0` ，运算如下：

    .. math::
        \text{output}(N_i, C_j, l) = \frac{1}{l_{ker}} \sum_{n=0}^{l_{ker}-1}
        \text{input}(N_i, C_j, s_0 \times l + n)

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入shape为 :math:`(N, C_{in}, L_{in})` 的Tensor。
        - **kernel_size** (Union(int, tuple[int])) - 指定池化核尺寸大小。
        - **stride** (Union(int, tuple[int]), 可选) - 池化操作的移动步长。可以是一个整数或者包含一个整数的tuple。默认值： ``None``，表示与 ``kernel_size`` 值相同。
        - **padding** (Union(int, tuple[int]), 可选) - 池化填充长度。可以是一个整数或者包含一个整数的tuple。默认值： ``0`` 。
        - **ceil_mode** (bool, 可选) - 如果为 ``True`` ，用ceil代替floor来计算输出的shape。默认值： ``False`` 。
        - **count_include_pad** (bool, 可选) - 如果为 ``True`` ，平均计算将包括零填充。默认值： ``True`` 。

    返回：
        Tensor，shape为 :math:`(N, C_{in}, L_{out})` 。

    异常：
        - **TypeError** - `input` 不是一个Tensor。
        - **TypeError** - `kernel_size` 或 `stride` 不是int，或不是长度为1的tuple。
        - **TypeError** - `ceil_mode` 或 `count_include_pad` 不是bool。
        - **ValueError** - `kernel_size` 或 `stride` 小于1。
        - **ValueError** - `kernel_size`、`stride` 或 `padding` 不是int，或不是长度为1的tuple。
