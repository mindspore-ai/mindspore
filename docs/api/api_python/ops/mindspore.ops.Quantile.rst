mindspore.ops.Quantile
======================

.. py:class:: mindspore.ops.Quantile(dim=None, keep_dims=False, ignore_nan=False)

    计算 `input` 中所有元素的第 `q` 分位数，如果第 `q` 分位数位于两个数据点之间，则返回使用线性插值计算结果。

    更多参考详见 :func:`mindspore.ops.quantile` 和 :func:`mindspore.ops.nanquantile`。

    参数：
        - **dim** (int，可选) - 要减少的维度。默认情况下 `axis` 为None，导致输入Tensor在计算前被展平。默认值： ``None`` 。
        - **keep_dims** (bool，可选) - 输出Tensor是否保留维度。默认值： ``False`` 。
        - **ignore_nan** (bool，可选) - 是否忽略输入中的NaN值。默认值： ``False`` 。

    输入：
        - **input** (Tensor) - 输入Tensor。支持的数据类型为：float32、float64。
        - **q** (Union[float, Tensor]) - 标量或1D Tensor。其值范围在[0, 1]，支持的数据类型为：float32、float64。

    输出：
        输入Tensor，数据类型与 `input` 一致。
