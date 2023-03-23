mindspore.ops.CompareAndBitpack
================================

.. py:class:: mindspore.ops.CompareAndBitpack

    将 `x` 的值与 `threshold` 进行比较，并将比较结果作为二进制数转换为uint8格式。
    每次比较返回一个布尔值，如果 `x` > `threshold` ，则为True，否则为False。

    若 `x` 的shape为 :math:`(s_0, s_1, ..., s_n)` ，则输出的shape为 :math:`(s_0, s_1, ..., s_n / 8)` 。

    输入：
        - **x** (Tensor) - 输入Tensor。与 `threshold` 进行比较并二进制转换的值。数据类型必须为bool、float16、float32、float64、int8、int16、int32、int64之一。（注:目前， `x` 最里层的维数必须能被8整除。）
        - **threshold** (Tensor) - 0D Tensor，数据类型需与 `x` 相同。

    输出：
        Tensor，数据类型为uint8.

    异常：
        - **TypeError** - 输入 `x` 和 `threshold` 不是Tensor。
        - **TypeError** - 输入 `x` 的数据类型不是bool、float16、float32、float64、int8、int16、int32、int64之一。
        - **TypeError** - 输入 `threshold` 与 `x` 的数据类型不一致。
        - **ValueError** - 输入 `threshold` 不是一个0D的Tensor。
        - **ValueError** - 输入 `x` 不是一个0D的Tensor。
        - **ValueError** - 输入 `x` 最里层的维数不能被8整除。