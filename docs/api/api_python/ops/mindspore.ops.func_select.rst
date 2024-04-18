mindspore.ops.select
====================

.. py:function:: mindspore.ops.select(condition, input, other)

    根据条件判断Tensor中的元素的值，来决定输出中的相应元素是从 `input` （如果元素值为True）还是从 `other` （如果元素值为False）中选择。

    该算法可以被定义为：

    .. math::

        out_i = \begin{cases}
        input_i, & \text{if } condition_i \\
        other_i, & \text{otherwise}
        \end{cases}

    参数：
        - **condition** (Tensor[bool]) - 条件Tensor，决定选择哪一个元素，shape是 :math:`(x_1, x_2, ..., x_N, ..., x_R)`。
        - **input** (Union[Tensor, int, float]) - 第一个被选择的Tensor或者数字。
          如果input是一个Tensor，那么shape是或者可以被广播为 :math:`(x_1, x_2, ..., x_N, ..., x_R)`。
          如果input是int或者float，那么将会被转化为int32或者float32类型，并且被广播为与y相同的shape。x和y中至少要有一个Tensor。
        - **other** (Union[Tensor, int, float]) - 第二个被选择的Tensor或者数字。
          如果other是一个Tensor，那么shape是或者可以被广播为 :math:`(x_1, x_2, ..., x_N, ..., x_R)`。
          如果other是int或者float，那么将会被转化为int32或者float32类型，并且被广播为与x相同的shape。x和y中至少要有一个Tensor。

    返回：
        Tensor，与 `condition` 的shape相同。

    异常：
        - **TypeError** - `input` 和 `other` 不是Tensor、int或者float。
        - **ValueError** - 输入的shape不能被广播。
