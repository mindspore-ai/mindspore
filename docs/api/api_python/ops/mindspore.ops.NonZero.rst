mindspore.ops.NonZero
======================

.. py:class:: mindspore.ops.NonZero

    返回所有非零元素的索引位置。

    输入：
        - **input** (Tensor) - 输入Tensor，其秩应大于等于1。

    输出：
        二维Tensor，数据类型为int64，包含所有输入中的非零元素的索引位置。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **ValueError** - 如果 `input` 的维度等于0。
