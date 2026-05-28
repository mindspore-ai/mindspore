mindspore.mint.t
======================

.. py:function:: mindspore.mint.t(input)

    转置输入Tensor。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入Tensor。

    返回：
        Tensor，转置2维Tensor，一维Tensor按原样返回。

    异常：
        - **ValueError** - 输入Tensor维度大于2。
        - **ValueError** - 输入为空Tensor。
        - **TypeError** - 输入不是Tensor。
