mindspore.nn.AdaptiveMaxPool1d
==============================

.. py:class:: mindspore.nn.AdaptiveMaxPool1d(output_size)

    在一个输入Tensor上应用1D自适应最大池化运算，可被视为组成一个1D输入平面。

    通常，输入的shape为 :math:`(N_{in}, C_{in}, L_{in})` 或 :math:`(C_{in}, L_{in})` 。
    输出的shape为 :math:`(N_{in}, C_{in}, L_{out})` 或 :math:`(C_{in}, L_{out})` ，其中 :math:`L_{out}` 由 `output_size` 定义。

    .. note::
        Ascend平台不支持 `return_indices` 参数。

    参数：
        - **output_size** (int) - 目标输出大小 :math:`L_{out}` 。
        - **return_indices** (bool) - 如果为True，输出最大值的索引，默认值为False。

    输入：
        - **input** (Tensor) - 输入shape为 :math:`(N_{in}, C_{in}, L_{in})` 或 :math:`(C_{in}, L_{in})` ，数据类型为float16、float32。

    输出：
        Tensor，数据类型与 `input` 相同。
        输出的shape为 :math:`(N_{in}, C_{in}, L_{out})` 或 :math:`(C_{in}, L_{out})` 。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `output_size` 不是int类型。
        - **TypeError** - 如果 `return_indices` 不是bool类型。
        - **ValueError** - 如果 `output_size` 小于1。
        - **ValueError** - 如果 `input` 的维度不等于2或3。
