mindspore.nn.LossBase
======================

.. py:class:: mindspore.nn.LossBase(reduction='mean')

    损失函数的基类。

    自定义损失函数时应重写 `construct` ，并使用方法 `self.get_loss` 将 `reduction` 应用于loss计算。

    参数：
        - **reduction** (str) - 指定应用于输出结果的计算方式。可选值有："mean"、"sum"、"none"。默认值："mean"。

    异常：
        - **ValueError** - `reduction` 不为'none'、'mean'或'sum'。

    .. py:method:: get_axis(x)

        获取输入的轴范围。

        参数：
            - **x** (Tensor) - 任何shape的Tensor。

    .. py:method:: get_loss(x, weights=1.0)

        计算加权损失。

        参数：
            - **x** (Tensor) - shape为 :math:`(N, *)` 的输入Tensor，其中 :math:`*` 表示任意数量的附加维度。
            - **weights** (Union[float, Tensor]) - 可选值，要么rank为0，要么rank与输入相同，并且必须可广播到输入（即，所有维度必须为 `1` ，或与相应输入的维度相同）。默认值：1.0。

        返回：
            返回加权损失。
