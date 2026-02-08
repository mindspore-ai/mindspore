mindspore.Tensor.select
=======================

.. py:method:: mindspore.Tensor.select(dim, index) -> Tensor

    沿着选定的维度在给定的索引处进行切片。

    .. warning::
        这是一个实验性API，可能会更改或删除。

    参数：
        - **dim** (int) - 指定切片的维度。
        - **index** (int) - 指定索引值。

    返回：
        Tensor。

    .. py:method:: mindspore.Tensor.select(condition, y) -> Tensor
        :noindex:

    根据条件判断Tensor中的元素的值，来决定输出中的相应元素是从 `self` （如果元素值为True）还是从 `y` （如果元素值为False）中选择。

    该算法可以被定义为：

    .. math::
        out_i = \begin{cases}
        self_i, & \text{if } condition_i \\
        y_i, & \text{otherwise}
        \end{cases}

    参数：
        - **condition** (Tensor[bool]) - 条件Tensor，决定选择哪一个元素。维度是： :math:`(x_1, x_2, ..., x_N, ..., x_R)` 。
        - **y** (Union[Tensor, int, float]) - 备选的Tensor或者数字。

          - 如果 `y` 是一个Tensor，那么shape是或者可以被广播为： :math:`(x_1, x_2, ..., x_N, ..., x_R)`。
          - 如果 `y` 是int或者float，那么将会被转化为int32或者float32类型，并且被广播为与 `self` 相同的shape。 `self` 和 `y` 中至少要有一个Tensor。

    返回：
        Tensor。与 `condition` 的shape相同。

    异常：
        - **TypeError** - 如果 `y` 不是Tensor、int或者float。
        - **ValueError** - 输入的shape不能被广播。
