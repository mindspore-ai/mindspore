mindspore.mint.nn.ConstantPad1d
===============================

.. py:class:: mindspore.mint.nn.ConstantPad1d(padding, value)

    根据 `padding` 和 `value` 对输入tensor的最后一维进行填充。

    更多参考详见 :func:`mindspore.mint.nn.functional.pad`。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **padding** (Union[int, tuple[int], list[int]]) - 填充的大小。
            - 如果为int，则对所有边界进行相同大小的填充。
            - 如果为tuple或list，则顺序为 :math:`(pad_{left}, pad_{right})`。
        - **value** (Union[int, float]) - 指定填充值。

    输入：
        - **input** (Tensor) - 输入tensor，shape为 :math:`(N, *)`，其中 :math:`*` 表示任意附加维度。

    输出：
        Tensor，填充后的tensor。
