mindspore.ops.Bincount
======================

.. py:class:: mindspore.ops.Bincount

    计算整数数组中每个值的出现次数。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    输入：
        - **array** (Tensor) - int32数据类型的Tensor。
        - **size** (Tensor) - int32数据类型的非负Tensor。
        - **weights** (Tensor) - 与 `array` shape相同或维度为0的Tensor。当维度为0时，所有权重均等于1。数据类型必须是以下类型之一：int32、int64、float32、float64。

    输出：
        Tensor，数据类型与 `weights` 相同。

    异常：
        - **TypeError** - `array` 的数据类型不是int32。
        - **TypeError** - `size` 的数据类型不是int32。
        - **ValueError** - `size` 中的数据存在负值。
        - **ValueError** - `weights` 是空Tensor。
        - **ValueError** - `weights` 的size不为0且和 `array` 的shape不同。
        - **TypeError** - `weights` 的数据类型不是int32、int64、float32或float64。
