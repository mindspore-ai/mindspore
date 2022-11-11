mindspore.ops.Bucketize
=======================

.. py:class:: mindspore.ops.Bucketize(boundaries)

    根据 `boundaries` 对 `input` 进行桶化。

    参数：
        - **boundaries** (list[float]) - 浮点数的排序列表给出了存储桶的边界，没有默认值。

    输入：
        - **input** (Tensor) - 包含搜索值的Tensor。

    输出：
        Tensor，shape与 `input` 的shape相同，数据类型为int32。

    异常：
        - **TypeError** - `boundaries` 不是浮点型的列表。
        - **TypeError** - `input` 不是Tensor。
