mindspore.Tensor.narrow
=======================

.. py:method:: mindspore.Tensor.narrow(axis, start, length)

    沿指定轴，指定起始位置获取指定长度的Tensor。

    参数：
        - **axis** (int) - 指定的轴。
        - **start** (int) - 指定的起始位置。
        - **length** (int) - 指定的长度。

    返回：
        Tensor。

    异常：
        - **TypeError** - axis不是int类型。
        - **TypeError** - start不是int类型。
        - **TypeError** - length不是int类型。
        - **ValueError** - axis取值不在[0, ndim-1]范围内。
        - **ValueError** - start取值不在[0, shape[axis]-1]范围内。
        - **ValueError** - start+length超出Tensor的维度范围shape[axis]-1。