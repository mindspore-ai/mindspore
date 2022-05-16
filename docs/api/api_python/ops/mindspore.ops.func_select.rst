mindspore.ops.select
====================

.. py:function:: mindspore.ops.select(condition, x, y)

    根据条件判断Tensor中的元素的值来，决定输出中的相应元素是从 `x`（如果元素值为True）还是从 `y`（如果元素值为False）中选择。

    该算法可以被定义为：

    .. math::

        out_i = \begin{cases}
        x_i, & \text{if } condition_i \\
        y_i, & \text{otherwise}
        \end{cases}

    **参数：**

    - **condition** (Tensor[bool]) - 条件Tensor， 决定选择哪一个元素，shape是 :math:`(x_1, x_2, ..., x_N, ..., x_R)`。
    - **x** (Union[Tensor, int, float]) - 第一个被选择的Tensor或者数字。
      如果x是一个Tensor，那么shape是 :math:`(x_1, x_2, ..., x_N, ..., x_R)`。
      如果x是int或者float， 那么将会被转化为int32或者float32类型，并且被广播为与y相同的shape。x和y中至少要有一个Tensor。
    - **y** (Union[Tensor, int, float]) - 第二个被选择的Tensor或者数字。
      如果y是一个Tensor，那么shape是 :math:`(x_1, x_2, ..., x_N, ..., x_R)`。
      如果y是int或者float， 那么将会被转化为int32或者float32类型，并且被广播为与x相同的shape。x和y中至少要有一个Tensor。


    **返回：**

    Tensor，与 `condition` 的shape相同。

    **异常：**

    - **TypeError** - `x` 和 `y` 不是Tensor、int或者float。
    - **TypeError** - 输入的shape不同。
