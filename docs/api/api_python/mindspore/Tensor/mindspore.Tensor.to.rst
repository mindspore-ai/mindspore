mindspore.Tensor.to
===================

.. py:method:: mindspore.Tensor.to(dtype=None, non_blocking=False, copy=False) -> Tensor

    转换Tensor的数据类型。

    .. note::
        - 当将复数转换为布尔类型时，复数的虚部不被考虑。只要实部非零，它就返回True；否则，返回False。
        - `non_blocking` 和 `copy` 在静态图或者jit内部不生效。

    参数：
        - **dtype** (dtype.Number, 可选) - 输出Tensor的有效数据类型，只允许常量值。bool类型仅在PyNative模式下支持。默认值是 ``None`` 。
        - **non_blocking** (bool, 可选) - 数据异步转换。如果是 ``True`` ，数据类型异步转换。如果是 ``False`` ，数据同步转换。默认值是 ``False`` 。
        - **copy** (bool, 可选) - 当 `copy` 设置为 ``True`` 时，即使Tensor已经符合所需的转换，也会创建一个新的Tensor。默认值是 ``False`` 。

    返回：
        Tensor，其数据类型为 `dtype` 。


    .. py:method:: mindspore.Tensor.to(device=None, dtype=None, non_blocking=False, copy=False) -> Tensor
        :noindex:

    将Tensor的设备和数据类型转换成指定的 `device` 和 `dtype` 。

    .. note::
        `device`、 `non_blocking` 和 `copy` 在静态图或者jit内部不生效。

    参数：
        - **device** (str, 可选) - 用于指定输出Tensor所在的硬件设备。默认是 ``None`` 。
        - **dtype** (dtype.Number, 可选) - 输出Tensor的有效数据类型，只允许常量值。bool类型仅在PyNative模式下支持。默认值是None。
        - **non_blocking** (bool, 可选) - 数据异步转换。如果是 ``True`` ，数据类型异步转换。如果是 ``False`` ，数据同步转换。默认值是 ``False`` 。
        - **copy** (bool, 可选) - 当 `copy` 设置为 ``True`` 时，即使Tensor已经符合所需的转换，也会创建一个新的Tensor。默认值是 ``False`` 。

    返回：
        Tensor，其所在的设备为 `device` ，其数据类型为 `dtype` 。

    .. py:method:: mindspore.Tensor.to(other, non_blocking=False, copy=False) -> Tensor
        :noindex:

    转换Tensor的设备和数据类型，转换后的Tensor和 `other` 保持相同的设备和数据类型。

    .. note::
        `non_blocking` 和 `copy` 在静态图或者jit内部不生效。

    参数：
        - **other** (Tensor) - 输出Tensor的 `device` 和 `dtype` 需要和 `other` 保持一致。
        - **non_blocking** (bool, 可选) - 数据异步转换。如果是 ``True`` ，数据类型异步转换。如果是 ``False`` ，数据同步转换。默认值是 ``False`` 。
        - **copy** (bool, 可选) - 当 `copy` 设置为 ``True`` 时，即使Tensor已经符合所需的转换，也会创建一个新的Tensor。默认值是 ``False`` 。

    返回：
        Tensor，其所在的设备和数据类型需要和 `other` 保持一致。
