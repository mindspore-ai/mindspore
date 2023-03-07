mindspore.ops.view_as_real
==========================

.. py:function:: mindspore.ops.view_as_real(input)

    将复数Tensor看作实数Tensor。返回的实数Tensor的最后一维大小为2，由复数的实部和虚部组成。

    参数：
        - **input** (Tensor) - 输入必须是一个复数Tensor。

    返回：
        实数Tensor。

    异常：
        - **TypeError** - 输入Tensor不是复数类型。
