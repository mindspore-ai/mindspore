mindspore.ops.rfft
=================================

.. py:function:: mindspore.ops.rfft(input, n=None, dim=-1, norm=None)

    计算实数输入的1-D离散傅里叶变换。

    .. note::
        - `rfft` 目前仅用于 `mindscience` 科学计算场景，尚不支持其他使用场景。
        - `rfft` 尚不支持Windows平台。

    参数：
        - **input** (Tensor) - 输入的Tensor。
        - **n** (int, 可选) - 指定轴 `dim` 上用于计算的元素数量。如果给定 `n` ，输入会被截断或补零至 `dim` 轴大小为 `n` 。
          默认值： ``None`` ，默认为 `input.shape[dim]`。
        - **dim** (int, 可选) - 指定需变换的轴。默认值： ``None`` ，默认为 ``-1`` 。
        - **norm** (str, 可选) - 标准化类型。默认值： ``None`` ，默认为 ``"backward"`` 。
          3种标准化类型定义为：

            - ``"backward"`` ：不乘以标准化系数。
            - ``"forward"`` ： 乘以标准化系数 :math:`1/n` 。
            - ``"ortho"`` ：乘以标准化系数 :math:`1/\sqrt{n}` 。

    返回：
        Tensor， `input` 进行实数离散傅里叶变换的结果，变换结果的数据类型为complex64/128， `dim` 轴大小为 :math:`n // 2 + 1`，
        其余轴大小与 `input` 相同。


    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `input` 数据类型不是int16，int32，int64，float32，float64。
        - **TypeError** - 如果 `n` 或 `dim` 不是int类型。
        - **ValueError** - 如果 `dim` 中的值超出： :math:`[-input.ndim, -input.ndim)` 范围。
        - **ValueError** - 如果 `n` 小于1。
        - **ValueError** - 如果 `norm` 的值不是 ``"backward"`` ， ``"forward"`` 或 ``"ortho"`` 。

