mindspore.ops.NanToNum
======================

.. py:class:: mindspore.ops.NanToNum(nan=0.0, posinf=None, neginf=None)

    将输入中的 `NaN` 、正无穷大和负无穷大值分别替换为 `nan` 、 `posinf` 和 `neginf` 指定的值。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多细节请参考 :func:`mindspore.ops.nan_to_num` 。

    参数：
        - **nan** (float，可选) - 替换 `NaN` 的值。默认值为 ``0.0`` 。
        - **posinf** (float，可选) - 如果是一个数字，则为替换正无穷的值。如果为 ``None`` ，则将正无穷替换为 `x` 类型支持的上限。默认值为 ``None`` 。
        - **neginf** (float，可选) - 如果是一个数字，则为替换负无穷的值。如果为 ``None`` ，则将负无穷替换为 `x` 类型支持的下限。默认值为 ``None`` 。

    输入：
        - **x** (Tensor) - 任意维度的输入Tensor。类型必须为float32或float16。

    输出：
        Tensor，数据shape和类型与 `input` 相同。
