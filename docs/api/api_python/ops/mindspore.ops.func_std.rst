mindspore.ops.std
=================

.. py:function:: mindspore.ops.std(input, axis=None, ddof=0, keepdims=False)

    计算Tensor在指定轴上的标准差。

    参数：
        - **input** (Tensor[Number]) - 输入Tensor。
        - **axis** (Union[int, tuple(int)]，可选) - 指定轴。如果为 ``None`` ，计算 `input` 中的所有元素。默认 ``None`` 。
        - **ddof** (Union[int, bool]，可选) - δ自由度。默认 ``0`` 。

          - 如果为整数，计算中使用的除数是 :math:`N - ddof` ，其中 :math:`N` 表示元素的数量。
          - 如果为bool值， ``True`` 与 ``False`` 分别对应 `ddof` 为整数时的 ``1`` 与 ``0`` 。
          - 如果取值为0、1、True或False，支持的平台只有 `Ascend` 和 `CPU` 。其他情况下，支持平台是 `Ascend` 、 `GPU` 和 `CPU` 。
        - **keepdims** (bool，可选) - 输出Tensor是否保留维度。默认 ``False`` 。

    返回：
        Tensor
