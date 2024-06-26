mindspore.ops.matmul
=====================

.. py:function:: mindspore.ops.matmul(input, other)

    计算两个数组的矩阵乘积。

    .. note::
        不支持NumPy参数 `out` 、 `casting` 、 `order` 、 `subok` 、 `signature` 、 `extobj` 。 `input` 和 `other` 的数据类型必须一致。在Ascend平台上，`input` 和 `other` 的秩必须在 1 到 6 之间。

    参数：
        - **input** (Tensor) - 输入Tensor，不支持Scalar， `input` 的最后一维度和 `other` 的倒数第二维度相等，且 `input` 和 `other` 彼此支持广播。
        - **other** (Tensor) - 输入Tensor，不支持Scalar， `input` 的最后一维度和 `other` 的倒数第二维度相等，且 `input` 和 `other` 彼此支持广播。

    返回：
        Tensor或Scalar，输入的矩阵乘积。仅当 `input` 和 `other` 为一维向量时，输出为Scalar。

    异常：
        - **TypeError** - `input` 的dtype和 `other` 的dtype不一致。
        - **ValueError** -  `input` 的最后一维度和 `other` 的倒数第二维度不相等，或者输入的是Scalar。
        - **ValueError** - `input` 和 `other` 彼此不能广播。
        - **RuntimeError** - 在Ascend平台上， `input` 或 `other` 的秩小于 1 或大于 6。
