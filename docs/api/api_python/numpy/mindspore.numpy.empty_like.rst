mindspore.numpy.empty_like
=================================

.. py:function:: mindspore.numpy.empty_like(prototype, dtype=None, shape=None)

    返回一个shape和类型与给定数组相同的新数组。

    .. note::
        输入数组在整个维度上必须具有相同的大小。如果 ``prototype`` 不是Tensor，且未设置 ``dtype`` ，则 ``dtype`` 默认为float32。

    参数：
        - **prototype** (Union[Tensor, list, tuple]) - 原数组，其中生成数组的shape、类型默认和原数组相同。
        - **dtype** (mindspore.dtype, 可选) - 覆盖结果的数据类型。默认值： ``None`` 。
        - **shape** (int, ints的序列, 可选) - 覆盖结果的shape。默认值： ``None`` 。

    返回：
        Tensor，给定shape和类型与 ``prototype`` 相同的未初始化（任意）数据的数组。

    异常：
        - **ValueError** - 如果 ``prototype`` 不是Tensor、List或Tuple。