mindspore.Tensor.type
=====================

.. py:method:: mindspore.Tensor.type(dtype=None)

    将Tensor的dtype转换成 `dtype` 。如果 `dtype` 为 ``None`` ，则返回原类型。

    参数：
        - **dtype** (mindspore.dtype，可选) - 指定输出Tensor的数据类型。默认值： ``None`` 。

    返回：
        Tensor或str。如果 `dtype` 为 ``None`` 则返回str，str描述了Tensor的数据类型。如果 `dtype` 不为 ``None`` ，则返回Tensor，返回Tensor的dtype是 `dtype` 。
