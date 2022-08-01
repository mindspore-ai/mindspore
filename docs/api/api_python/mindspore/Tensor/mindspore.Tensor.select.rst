mindspore.Tensor.select
=======================

.. py:method:: mindspore.Tensor.select(condition, y)

    根据条件判断Tensor中的元素的值，来决定输出中的相应元素是从当前Tensor（如果元素值为True）还是从 `y` （如果元素值为False）中选择。

    该算法可以被定义为：

    .. math::

        out_i = \begin{cases}
        tensor_i, & \text{if } condition_i \\
        y_i, & \text{otherwise}
        \end{cases}

    参数：
        - **condition** (Tensor[bool]) - 条件Tensor，决定选择哪一个元素。shape与当前的Tensor相同。
        - **y** (Union[Tensor, int, float]) - 如果y是一个Tensor，那么shape与当前Tensor相同。如果y是int或者float，那么将会被转化为int32或者float32类型，并且被广播为与当前Tensor相同的shape。

    返回：
        Tensor，与当前Tensor的shape相同。

    异常：
        - **TypeError** - `y` 不是Tensor、int或者float。
        - **ValueError** - 输入的shape不相同。