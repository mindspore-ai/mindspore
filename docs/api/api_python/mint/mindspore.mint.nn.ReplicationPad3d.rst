mindspore.mint.nn.ReplicationPad3d
==================================

.. py:class:: mindspore.mint.nn.ReplicationPad3d(padding)

    使用输入边界值，对输入 `input` 的最后三维进行填充。

    更多参考详见 :func:`mindspore.mint.nn.functional.pad`。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **padding** (Union[int, tuple, list]) - 填充的大小。

    输入：
        - **input** (Tensor) - 输入tensor，shape为 :math:`(N, D_{in}, H_{in}, W_{in})` 或 :math:`(N, C, D_{in}, H_{in}, W_{in})`。

    输出：
        Tensor，shape为 :math:`(C, D_{out}, H_{out}, W_{out})` 或 :math:`(N, C, D_{out}, H_{out}, W_{out})`。其中:
        :math:`D_{out} = D_{in} + pad_{front} + pad_{back}` 
        :math:`H_{out} = H_{in} + pad_{up} + pad_{down}`
        :math:`W_{out} = W_{in} + pad_{left} + pad_{right}`
        
