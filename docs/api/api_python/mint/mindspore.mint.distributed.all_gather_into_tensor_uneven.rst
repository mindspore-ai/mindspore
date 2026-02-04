mindspore.mint.distributed.all_gather_into_tensor_uneven
========================================================

.. py:function:: mindspore.mint.distributed.all_gather_into_tensor_uneven(output, input, output_split_sizes=None, group=None, async_op=False)

    收集并拼接各设备上的张量，各设备上的张量第一维可以不一致。

    .. note::
        - 各设备的输入张量除第一个维度外必须具有相同shape。
        - 输出张量的第一个维度是所有设备输入张量第一个维度之和。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **output** (Tensor) - 拼接后的输出张量，shape为 :math:`(\sum_{i=0}^{N-1} x_{i1}, x_2, ..., x_R)`，其中N为通信组中的设备数量。
        - **input** (Tensor) - 本地输入张量，shape为 :math:`(x_{k1}, x_2, ..., x_R)`，k表示当前设备rank。
        - **output_split_sizes** (list[int], 可选) - 指定各设备输入的第一个维度尺寸。当提供时必须与实际输入尺寸匹配。当为None时，将会在所有设备上进行平均分配。默认值： ``None``。
        - **group** (str, 可选) - 通信组标识符。None表示使用默认通信组。默认值： ``None``。
        - **async_op** (bool, 可选) - 是否启用异步操作。默认值： ``False``。

    返回：
        CommHandle。若 `async_op` 是True，则CommHandle是一个异步工作句柄；若 `async_op` 是False，则CommHandle将返回None。

    异常：
        - **ValueError** - 如果 `input` 的shape与 `output_split_sizes` 的值不满足约束。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
