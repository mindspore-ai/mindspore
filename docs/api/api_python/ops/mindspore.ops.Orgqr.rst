mindspore.ops.Orgqr
====================

.. py:class:: mindspore.ops.Orgqr

    计算 :class:`mindspore.ops.Geqrf` 返回的正交矩阵 :math:`Q` 的显式表示。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多细节请参考 :func:`mindspore.ops.orgqr` 。

    输入：
        - **x** (Tensor) - shape :math:`(*, M, N)` 的Tensor，表示二维或者三维矩阵。数据类型为float32、float64、complex64或者complex128。
        - **tau** (Tensor) - Householder转换的反射系数，其shape为 :math:`(*, K)` ，其中 `K` 小于等于 `N` 。数据类型与 `x` 一致。

    输出：
        Tensor，数据类型与shape与 `input` 一致。
