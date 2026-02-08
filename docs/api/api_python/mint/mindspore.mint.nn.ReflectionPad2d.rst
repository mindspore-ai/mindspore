mindspore.mint.nn.ReflectionPad2d
=================================

.. py:class:: mindspore.mint.nn.ReflectionPad2d(padding)

    使用输入边界的反射，对输入 `input` 的最后二维进行填充。

    更多参考详见 :func:`mindspore.mint.nn.functional.pad`。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **padding** (Union[int, tuple, list]) - 指定填充的大小。

    输入：
        - **input** (Tensor) - 输入tensor，shape为 :math:`(C, H_{in}, W_{in})` 或 :math:`(N, C, H_{in}, W_{in})`。

    输出：
        Tensor，填充后的tensor。
