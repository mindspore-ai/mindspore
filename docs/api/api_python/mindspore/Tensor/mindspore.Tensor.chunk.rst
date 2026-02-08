mindspore.Tensor.chunk
======================

.. py:method:: mindspore.Tensor.chunk(chunks, dim=0) -> tuple[Tensor]

    沿着指定轴 `dim` 将输入Tensor切分成 `chunks` 个sub-tensor。

    .. note::
        此函数返回的sub-tensor数量可能小于通过 `chunks` 指定的sub-tensor数量。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **chunks** (int) - 要切分的sub-tensor数量。
        - **dim** (int，可选) - 指定需要分割的维度。默认值： ``0`` 。

    返回：
        tuple[Tensor]。

    异常：
        - **TypeError** - `dim` 不是int类型。
        - **TypeError** - `chunks` 不是int。
        - **ValueError** - 参数 `dim` 超出 :math:`[-self.ndim, self.ndim)` 范围。
        - **ValueError** - 参数 `chunks` 不是正数。

    .. py:method:: mindspore.Tensor.chunk(chunks, axis=0) -> tuple[Tensor]
        :noindex:

    沿着指定轴 `axis` 将输入Tensor切分成 `chunks` 个sub-tensor。

    .. note::
        此函数返回的数量可能小于通过 `chunks` 指定的数量。

    参数：
        - **chunks** (int) - 要切分的sub-tensor数量。
        - **axis** (int，可选) - 指定需要分割的维度。默认值： ``0`` 。

    返回：
        tuple[Tensor]。

    异常：
        - **TypeError** - `axis` 不是int类型。
        - **TypeError** - `chunks` 不是int。
        - **ValueError** - 参数 `axis` 超出 :math:`[-self.ndim, self.ndim)` 范围。
        - **ValueError** - 参数 `chunks` 不是正数。
