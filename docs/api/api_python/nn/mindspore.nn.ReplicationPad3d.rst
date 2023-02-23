mindspore.nn.ReplicationPad3d
=============================

.. py:class:: mindspore.nn.ReplicationPad3d(padding)

    根据 `padding` 对输入 `x` 的DHW维度上进行填充。

    参数：
        - **padding** (union[int, tuple]) - 填充 `x` 最后三个维度的大小。

          - 如果输入为int，则对所有边界进行相同大小的填充。
          - 如果是tuple，则顺序为 :math:`(pad_{left}, pad_{right}, pad_{up}, pad_{down}, pad_{front}, pad_{back})`。

    输入：
        - **x** (Tensor) - 维度为4D或5D的Tensor，shape为 :math:`(C, D_{in}, H_{in}, W_{in})` 或 :math:`(N, C, D_{in}, H_{in}, W_{in})` 。

    输出：
        Tensor，填充后的Tensor，shape为 :math:`(C, D_{out}, H_{out}, W_{out})`或 :math:`(N, C, D_{out}, H_{out}, W_{out})`。
        其中 :math:`D_{out} = D_{in} + pad_{front} + pad_{back}`, :math:`H_{out} = H_{in} + pad_{up} + pad_{down}`, :math:`W_{out} = W_{in} + pad_{left} + pad_{right}`。

    异常：
        - **TypeError** - `padding` 不是tuple或int。
        - **TypeError** - `padding` 中存在不是int的元素。
        - **ValueError** - `padding` 是tuple，且长度不能被2整除。
        - **ValueError** - `padding` 是tuple，且长度和Tensor的维度不匹配。
