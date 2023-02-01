mindspore.ops.kaiser_window
============================

.. py:function:: mindspore.ops.kaiser_window(window_length, periodic=True, beta=12.0)

    生成一个Kaiser window，也叫做Kaiser-Bessel window。

    Kaiser window定义：

    .. math::
        w(n) = \frac{I_{0}\left( \beta\sqrt{1 - \frac{4n^{2}}{(M - 1)^{2}}} \right)}{I_{0}(\beta)}

    n的范围为

    .. math::
        - \frac{M - 1}{2} \leq n \leq \frac{M - 1}{2}

    其中 :math:`I_0` 为零阶修正Bessel函数。

    参数：
        - **window_length** (int) - 输出window的大小。
        - **periodic** (bool, 可选) - 如果为True，则返回周期性window用于进行谱线分析。如果为False，则返回对称的window用于设计滤波器。默认值：True。
        - **beta** (float, 可选) - 形状参数，当 `beta` 变大时，窗口就会变窄。默认值：12.0。
    返回：
        Tensor，一个Kaiser window。

    异常：
        - **TypeError** - 如果 `window_length` 或 `beta` 不是整数。
        - **TypeError** - 如果 `periodic` 不是布尔类型。
        - **ValueError** - 如果 `window_length` 小于零。
