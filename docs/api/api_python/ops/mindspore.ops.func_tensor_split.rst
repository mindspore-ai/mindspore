mindspore.ops.tensor_split
===========================

.. py:function:: mindspore.ops.tensor_split(input, indices_or_sections, axis=0)

    根据指定索引或份数，将输入tensor拆分成多个子tensor。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：

        - 参数 `indices_or_sections` 将重命名为 `tensor_indices_or_sections` 。
        - 参数 `axis` 将重命名为 `dim` 。

        变更后的接口签名为 ``mindspore.ops.tensor_split(input, tensor_indices_or_sections, dim=0)`` 。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **indices_or_sections** (Union[int, tuple(int), list(int)]) - 指定索引或份数。

          - 如果是int类型，输入tensor将被拆分成 `indices_or_sections` 份。

            - 如果 :math:`input.shape[axis]` 能被 `indices_or_sections` 整除，那么子切片为相同大小 :math:`input.shape[axis] / n` 。
            - 如果 :math:`input.shape[axis]` 不能被 `indices_or_sections` 整除，那么前 :math:`input.shape[axis] \bmod n` 个切片的大小为 :math:`input.shape[axis] // n + 1` ，其余切片的大小为 :math:`input.shape[axis] // n` 。

          - 如果是tuple(int)或list(int)类型，则表示索引，输入tensor在索引处被拆分。
        - **axis** (int，可选) - `indices_or_sections` 所在的轴。默认 ``0`` 。

    返回：
        由多个tensor组成的tuple。
