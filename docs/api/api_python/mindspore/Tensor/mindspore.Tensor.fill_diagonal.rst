mindspore.Tensor.fill_diagonal
===============================

.. py:method:: mindspore.Tensor.fill_diagonal(fill_value, wrap=False)

    将 `self` Tensor的主对角线，填充成指定的值，并返回结果。 `self` 必须至少为2D，并且如果维度大于2，则需要所有维度上的长度均相等。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **fill_value** (float) - 对角线的填充值。
        - **wrap** (bool, 可选) - 控制对角线的元素是否继续在剩余的行上填充，如果 `self` 是高矩阵的话（矩阵的行比列多）。默认值：``False``。

    返回：
        - **y** (Tensor) - 和 `self` 具有相同的shape和数据类型。

    异常：
        - **TypeError** - 如果 `self` 的数据类型不是float32、int32、int64。
        - **ValueError** - 如果 `self` 的维度不大于1。
        - **ValueError** - 在 `self` 的维度大于2时，所有维度的长度不相等。
