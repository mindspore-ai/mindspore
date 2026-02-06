mindspore.ops.EuclideanNorm
============================

.. py:class:: mindspore.ops.EuclideanNorm(keep_dims=False)

    `ops.EuclideanNorm` 从2.8.0版本开始已被弃用，并将在未来版本中被移除。
    建议使用 :func:`mindspore.ops.norm` 且 `ord` 参数为 ``2`` 替代。

    计算Tensor沿指定轴的欧几里得范数（又称L2范数）。默认情况下会移除axis中指定的维度。

    参数：
        - **keep_dims** (bool，可选) - 如果为 ``True`` ，被归约的轴保留为1，如果为 ``False`` ，不保留给定的这些轴，默认值： ``False`` 。

    输入：
        - **x** (Tensor) - 参与归约计算的输入Tensor。
        - **axes** (Tensor) - 指定进行归约计算的维度。数据类型为：int32、int64。取值范围为： :math:`[-rank(x), rank(x))` 。

    输出：
        Tensor，与输入 `x` 具有相同的数据类型。

    异常：
        - **TypeError** - 如果 `keep_dims` 不是一个bool值。
        - **TypeError** - 如果 `x` 不是一个Tensor。
        - **ValueError** - 如果 `axes` 超出取值范围。
