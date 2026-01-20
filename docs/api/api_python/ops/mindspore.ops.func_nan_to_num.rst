mindspore.ops.nan_to_num
=========================

.. py:function:: mindspore.ops.nan_to_num(input, nan=None, posinf=None, neginf=None)

    将 `input` 中的 `NaN` 、正无穷大和负无穷大值分别替换为 `nan` 、 `posinf` 和 `neginf` 指定的值。

    .. warning::
        对于Ascend，仅支持 Atlas A2 训练系列产品。
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **nan** (number，可选) - `NaN` 的替换值，默认 ``None`` 。
        - **posinf** (number，可选) - `posinf` 的替换值。如果为 ``None`` ，则为 `input` 类型支持的上限，默认 ``None`` 。
        - **neginf** (number，可选) - `neginf` 的替换值。如果为 ``None`` ，则为 `input` 类型支持的下限，默认 ``None`` 。

    返回：
        Tensor
