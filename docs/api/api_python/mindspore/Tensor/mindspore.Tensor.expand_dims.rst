mindspore.Tensor.expand_dims
============================

.. py:method:: mindspore.Tensor.expand_dims(axis)

    沿指定轴扩展Tensor维度。

    参数：
        - **axis** (int) - 扩展维度指定的轴。

    返回：
        Tensor，指定轴上扩展的维度为1。

    异常：
        - **TypeError** - axis不是int类型。
        - **ValueError** - axis的取值不在[-self.ndim - 1, self.ndim + 1)。