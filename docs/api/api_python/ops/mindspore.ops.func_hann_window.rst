mindspore.ops.hann_window
==========================

.. py:function:: mindspore.ops.hann_window(window_length, periodic=True)

    生成一个Hann window。

    Hann window定义：

    .. math::
        w(n) = \frac{1}{2} - \frac{1}{2} \cos\left(\frac{2\pi{n}}{M-1}\right) \qquad 0 \leq n \leq M-1

    参数：
        - **window_length** (int) - 输出window的大小。
        - **periodic** (bool, 可选) - 如果为True，则返回周期性window用于进行谱线分析。如果为False，则返回对称的window用于设计滤波器。默认值：True。
    
    返回：
        Tensor，一个Hann window。

    异常：
        - **TypeError** - 如果 `window_length` 不是整数。
        - **TypeError** - 如果 `periodic` 不是布尔类型。
        - **ValueError** - 如果 `window_length` 小于零。
