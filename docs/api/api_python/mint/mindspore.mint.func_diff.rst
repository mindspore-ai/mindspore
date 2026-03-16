mindspore.mint.diff
=====================

.. py:function:: mindspore.mint.diff(input, n=1, dim=-1, prepend=None, append=None)

    计算沿给定维度的第n个正向差。

    第一个正向差通过 :math:`out[i] = input[i + 1] - input[i]` 计算得到，对于第n个正向差，可以通过递归使用 `diff` 得到。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 用于计算正向差的张量。
        - **n** (int，可选) - 递归计算正向差的次数。默认值为 ``1``。
        - **dim** (int，可选) - 指定用于计算正向差的维度。默认值为 ``-1``，即最后一个维度。
        - **prepend** (Tensor，可选) - 在计算正向差前，在 `input` 的 `dim` 维度上添加相应的值。 `prepend` 的维度需要与 `input` 保持一致，并且除了指定 `dim`，其余维度上， `prepend` 的大小需要与 `input` 保持一致。默认值为 ``None``。
        - **append** (Tensor，可选) - 在计算正向差前，在 `input` 的 `dim` 维度上追加相应的值。 `append` 的维度需要与 `input` 保持一致，并且除了指定 `dim`，其余维度上， `append` 的大小需要与 `input` 保持一致。默认值为 ``None``。

    返回：
        Tensor。计算得到的第n个正向差。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `n` 不是标量或标量Tensor。
        - **TypeError** - 如果 `dim` 不是标量或标量Tensor。
        - **TypeError** - 如果 `input` 的类型为complex64、complex128、float64、int16。