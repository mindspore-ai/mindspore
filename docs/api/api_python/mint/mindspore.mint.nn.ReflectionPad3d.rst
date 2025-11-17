mindspore.mint.nn.ReflectionPad3d
=================================

.. py:class:: mindspore.mint.nn.ReflectionPad3d(padding)

    使用输入边界的反射，对输入 `input` 的最后三维进行填充。

    更多参考详见 :func:`mindspore.mint.nn.functional.pad`。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **padding** (Union[int, tuple, list]) - 指定填充的大小。
            - 如果为int，则对所有边界进行相同大小的填充。
            - 如果为tuple或list，则顺序为 :math:`(pad_{front}, pad_{back}, pad_{up}, pad_{down}, pad_{left}, pad_{right})`。

    输入：
        - **input** (Tensor) - 输入tensor，shape为 :math:`(N, D_{in}, H_{in}, W_{in})` 或 :math:`(N, C, D_{in}, H_{in}, W_{in})`。

    输出：
        Tensor，填充后的tensor。
