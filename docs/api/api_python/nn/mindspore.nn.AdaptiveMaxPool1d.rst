mindspore.nn.AdaptiveMaxPool1d
==============================

.. py:class:: mindspore.nn.AdaptiveMaxPool1d(output_size)

    在一个输入Tensor上应用1D自适应最大池化运算，可被视为组成一个1D输入平面。

    通常，输入的shape为 :math:`(N_{in}, C_{in}, L_{in})` ，AdaptiveMaxPool1d在 :math:`L_{in}` 维度上计算区域最大值。
    输出的shape为 :math:`(N_{in}, C_{in}, L_{out})` ，其中， :math:`L_{out}` 为 `output_size`。

    .. note::
        :math:`L_{in}` 必须能被 `output_size` 整除。

    参数：
        - **output_size** (int) - 目标输出大小 :math:`L_{out}`。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C_{in}, L_{in})` 的Tensor，数据类型为float16或float32。

    输出：
        Tensor，其shape为 :math:`(N, C_{in}, L_{out})`，数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是float16或float32。
        - **TypeError** - `output_size` 不是int。
        - **ValueError** - `output_size` 小于1。
        - **ValueError** -  `x` 的最后一个维度小于 `output_size`。
        - **ValueError** -  `x` 的最后一个维度不能被 `output_size` 整除。
        - **ValueError** -  `x` 的shape长度不等于3。
