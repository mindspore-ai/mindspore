mindspore.ops.is_nonzero
========================

.. py:function:: mindspore.ops.is_nonzero(input)

    判断输入Tensor是否包含0或False，输入只能是单元素。

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Bool，如果输入Tensor包含一个单元素0或者一个单元素False，则返回False，否则返回True。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **ValueError** - `input` 的元素数量不等于1。