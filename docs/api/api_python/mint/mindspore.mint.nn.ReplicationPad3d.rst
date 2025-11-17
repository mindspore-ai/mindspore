mindspore.mint.nn.ReplicationPad3d
==================================

.. py:class:: mindspore.mint.nn.ReplicationPad3d(padding)

    使用输入边界值，对输入 `input` 的最后三维进行填充。

    更多参考详见 :func:`mindspore.mint.nn.functional.pad`。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **padding** (Union[int, tuple, list]) - 指定填充的大小。

    输入：
        - **input** (Tensor) - 输入 tensor，shape为 :math:`(N, D_{in}, H_{in}, W_{in})` 或 :math:`(N, C, D_{in}, H_{in}, W_{in})`，其中 `N` 是批处理大小，`C` 是通道数，`D_{in}` 是特征深度，`H_{in}` 是特征高度，`W_{in}` 是特征宽度。

    输出：
        Tensor，填充后的 tensor。

    异常：
        - **ValueError** - `padding` 是tuple或list，且长度和 tensor 的维度不匹配。
