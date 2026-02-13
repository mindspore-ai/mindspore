mindspore.nn.ReflectionPad3d
============================

.. py:class:: mindspore.nn.ReflectionPad3d(padding)

    使用反射的方式，以 `input` 的边界为对称轴，对 `input` 进行填充。3d代表填充的维度为三维。

    .. note::
        ReflectionPad3d尚不支持5D Tensor输入。

    参数：
        - **padding** (union[int, tuple]) - 填充大小，如果输入为int，则对所有边界进行相同大小的填充；如果是tuple，则顺序为 :math:`(\text{padding_left}, \text{padding_right}, \text{padding_top}, \text{padding_bottom}, \text{padding_front}, \text{padding_back})`。

    输入：
        - **x** (Tensor) - 4D Tensor，shape为 :math:`(N, D_{in}, H_{in}, W_{in})` 。

    输出：
        Tensor，填充后的Tensor，shape为 :math:`(N, D_{out}, H_{out}, W_{out})`。其中 :math:`D_{out} = D_{in} + \text{padding_front} + \text{padding_back}`、:math:`H_{out} = H_{in} + \text{padding_top} + \text{padding_bottom}` 、:math:`W_{out} = W_{in} +\text{padding_left} + \text{padding_right}`。

    异常：
        - **TypeError** - `padding` 不是tuple或int。
        - **TypeError** - `padding` 中存在不是int的元素。
        - **ValueError** - `padding` 是tuple，且长度不能被2整除。
        - **ValueError** - `padding` 是tuple，且存在负数。
        - **ValueError** - `padding` 是tuple，且长度和Tensor的维度不匹配。
