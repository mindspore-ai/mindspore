mindspore.ops.Select
=========================

.. py:class:: mindspore.ops.Select

    根据条件判断Tensor中的元素的值，决定输出中的相应元素是从 `x` （如果元素值为True）还是从 `y` （如果元素值为False）中选择。

    该算法可以被定义为：

    .. math::

        out_i = \begin{cases}
        x_i, & \text{if } condition_i \\
        y_i, & \text{otherwise}
        \end{cases}

    输入：
        - **condition** (Tensor[bool]) - 条件Tensor，决定选择哪一个元素，shape是 :math:`(x_1, x_2, ..., x_N, ..., x_R)`。
        - **x** (Tensor) - 第一个被选择的Tensor，shape是 :math:`(x_1, x_2, ..., x_N, ..., x_R)`。
        - **y** (Tensor) - 第二个被选择的Tensor，shape是 :math:`(x_1, x_2, ..., x_N, ..., x_R)`。

    输出：
        Tensor，具有与输入 `condition` 相同的shape。

    异常：
        - **TypeError** - 如果 `x` 或者 `y` 不是Tensor。
        - **ValueError** - 如果 `x` 的shape与 `y` 或者 `condition` 的shape不一致。