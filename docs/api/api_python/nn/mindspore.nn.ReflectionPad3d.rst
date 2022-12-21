mindspore.nn.ReflectionPad3d
============================

.. py:class:: mindspore.nn.ReflectionPad3d(padding)

    根据 `padding` 对输入 `x` 进行填充。

    参数：
        - **padding** (union[int, tuple]) - 填充大小，如果输入为int，则对所有边界进行相同大小的填充；如果是tuple，则顺序为 :math:`(pad_{left}, pad_{right}, pad_{up}, pad_{down}, pad_{front}, pad_{back})`。

    .. note::
        ReflectionPad3d尚不支持5D Tensor输入。

    输入：
        - **x** (Tensor) - 4D Tensor, shape为 :math:`(N, D_{in}, H_{in}, W_{in})` 。

    输出：
        Tensor，填充后的Tensor, shape为 :math:`(N, D_{out}, H_{out}, W_{out})`。其中 :math:`H_{out} = H_{in} + pad_{up} + pad_{down}`, :math:`W_{out} = W_{in} + pad_{left} + pad_{right}`, :math:`D_{out} = D_{in} + pad_{front} + pad_{back}` 。

    异常：
        - **TypeError** - `padding` 不是tuple或int。
        - **TypeError** - `padding` 中存在不是int的元素。
        - **ValueError** - `padding` 是tuple，且长度不能被2整除。
        - **ValueError** - `padding` 是tuple，且存在负数。
        - **ValueError** - `padding` 是tuple，且长度和Tensor的维度不匹配。
