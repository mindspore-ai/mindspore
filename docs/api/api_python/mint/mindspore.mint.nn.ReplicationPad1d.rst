mindspore.mint.nn.ReplicationPad1d
==================================

.. py:class:: mindspore.mint.nn.ReplicationPad1d(padding)

    使用输入边界值，对输入 `input` 的最后一维进行填充。

    更多参考详见 :func:`mindspore.mint.nn.functional.pad`。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **padding** (Union[int, tuple, list]) - 填充的大小。

          - 如果为int，则对所有边界进行相同大小的填充。
          - 如果为tuple或list，则顺序为 :math:`(\text{padding_left}, \text{padding_right})`。

    输入：
        - **input** (Tensor) - 输入tensor，shape为 :math:`(C, W_{in})` 或 :math:`(N, C, W_{in})`。

    输出：
        Tensor，填充后的tensor，shape为 :math:`(C, W_{out})` 或 :math:`(N, C, W_{out})`。其中：

        :math:`W_{out} = W_{in} + \text{padding_left} + \text{padding_right}`
