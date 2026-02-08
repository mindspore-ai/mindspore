mindspore.mint.nn.AdaptiveMaxPool1d
====================================

.. py:class:: mindspore.mint.nn.AdaptiveMaxPool1d(output_size, return_indices=False)

    对由多个输入平面组成的输入信号应用1D自适应最大池化。

    对于任何输入大小，输出大小为 :math:`L_{out}` 。
    输出特征的数量等于输入平面的数量。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. note::
        Atlas训练系列产品不支持反向传播。

    参数：
        - **output_size** (Union[int, tuple]) - 目标输出的size :math:`L_{out}` 。
        - **return_indices** (bool，可选) - 如果为 ``True`` ，输出最大值的索引，默认值为 ``False`` 。

    输入：
        - **input** (Tensor) - 输入特征的shape为 :math:`(N, C, L_{in})` 或  :math:`(C, L_{in})` 。

    输出：
        Union(Tensor, tuple(Tensor, Tensor))。

        - 如果 `return_indices` 为 ``False`` ，返回Tensor, 其shape为 :math:`(N, C_{in}, L_{out})`，数据类型与 `input` 相同。
        - 如果 `return_indices` 为 ``True``，则是一个包含了两个Tensor的Tuple，表示计算结果以及生成max值的位置。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 中的数据不是float16、float32、float64。
        - **TypeError** - `output_size` 不是int或者tuple。
        - **TypeError** - `return_indices` 不是bool。
        - **ValueError** - `output_size` 是tuple，但大小不是1。
