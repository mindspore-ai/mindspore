mindspore.mint.nn.ReflectionPad1d
=================================

.. py:class:: mindspore.mint.nn.ReflectionPad1d(padding)

    使用输入边界的反射，对输入 `input` 的最后一维进行填充。

    更多参考详见 :func:`mindspore.mint.nn.functional.pad`。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **padding** (Union[int, tuple, list]) - 指定填充的大小。
            - 如果为int，则对所有边界进行相同大小的填充。
            - 如果为tuple或list，则顺序为 :math:`(pad_{left}, pad_{right})`。

    输入：
        - **input** (Tensor) - 输入tensor，shape为 :math:`(C, W_{in})` 或 :math:`(N, C, W_{in})`，其中 `N` 是批处理大小，`C` 是通道数，`W_{in}` 是输入宽度。

    输出：
        Tensor，填充后的tensor。

