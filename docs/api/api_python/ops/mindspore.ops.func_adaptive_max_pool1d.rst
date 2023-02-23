mindspore.ops.adaptive_max_pool1d
=================================

.. py:function:: mindspore.ops.adaptive_max_pool1d(input, output_size, return_indices=False)

    对可以看作是由一系列1D平面组成的输入Tensor，应用一维自适应最大池化操作。

    通常，输入的shape为 :math:`(N_{in}, C_{in}, L_{in})` 或 :math:`(C_{in}, L_{in})` 。
    输出的shape为 :math:`(N_{in}, C_{in}, L_{out})` 或 :math:`(C_{in}, L_{out})` ，其中 :math:`L_{out}` 由 `output_size` 定义。

    .. note::
        Ascend平台不支持 `return_indices` 参数。

    参数：
        - **input** (Tensor) - 输入shape为 :math:`(N_{in}, C_{in}, L_{in})` 或 :math:`(C_{in}, L_{in})` ，数据类型为float16、float32。
        - **output_size** (int) - 大小为 :math:`L_{out}` 。
        - **return_indices** (bool) - 如果为True，输出最大值的索引，默认值为False。

    返回：
        Tensor，数据类型与 `input` 相同。
        输出的shape为 :math:`(N_{in}, C_{in}, L_{out})` 或 :math:`(C_{in}, L_{out})` 。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `output_size` 不是int类型。
        - **TypeError** - 如果 `return_indices` 不是bool类型。
        - **ValueError** - 如果 `output_size` 小于1。
        - **ValueError** - 如果 `input` 的维度不等于2或3。
