mindspore.ops.where
====================

.. py:function:: mindspore.ops.where(condition, x, y)

    返回一个Tensor，Tensor的元素从 `x` 或 `y` 中根据 `condition` 选择。

    .. math::

        output_i = \begin{cases} x_i,\quad &if\ condition_i \\ y_i,\quad &otherwise \end{cases}

    参数：
        - **condition** (Tensor[Bool]) - 如果是True，选取 `x` 中的元素，否则选取 `y` 中的元素。
        - **x** (Union[Tensor, Scalar]) - 在 `condition` 为True的索引处选择的值。
        - **y** (Union[Tensor, Scalar]) - 当 `condition` 为False的索引处选择的值。

    返回：
        Tensor，其中的元素从 `x` 和 `y` 中选取。

    异常：
        - **TypeError** - 如果 `condition` 不是Tensor。
        - **TypeError** - 如果 `x` 和 `y` 都是常量。
        - **ValueError** - `condition` 、 `x` 和 `y` 不能互相广播。
