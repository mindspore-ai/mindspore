mindspore.Tensor.astype
=======================

.. py:method:: mindspore.Tensor.astype(dtype, copy=True)

    将Tensor转为指定数据类型，可指定是否返回副本。

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：该接口将废弃，功能收敛至 ``to(dtype, non_blocking=False, copy=False) -> Tensor`` 。

    参数：
        - **dtype** (Union[mindspore.dtype, numpy.dtype, str]) - 指定的Tensor数据类型，可以是: `mindspore.dtype.float32` 、 `numpy.float32` 或 `float32` 的格式。
        - **copy** (bool, 可选) - 默认情况下，astype返回新拷贝的Tensor。如果该参数设为 ``False`` ，则返回输入Tensor而不是副本。默认值： ``True`` 。

    返回：
        Tensor，指定数据类型的Tensor。

    异常：
        - **TypeError** - 指定了无法解析的类型。
