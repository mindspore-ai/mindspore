mindspore.numpy.unique
=================================

.. py:function:: mindspore.numpy.unique(x, return_inverse=False)

    返回去重后的Tensor元素。当输入Tensor有多个维度时，将首先将其展平。

    .. note::
        不支持Numpy的 `axis` 、 `return_index` 和 `return_counts` 参数。在CPU上，该操作必须在图模式下执行。

    参数：
        - **x** (Tensor) - 要处理的输入Tensor。
        - **return_inverse** (bool，可选) - 如果为 ``True`` ，还返回唯一Tensor的索引。默认值： ``False`` 。

    返回：
        Tensor或Tensor的tuple。如果 `return_inverse` 为 ``False`` ，返回唯一Tensor；否则，返回Tensor的tuple。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。