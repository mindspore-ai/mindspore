mindspore.ops.HammingWindow
===========================

.. py:class:: mindspore.ops.HammingWindow(periodic=True, alpha=0.54, beta=0.46, dtype=mstype.float32)

    使用输入窗口长度计算汉明窗口函数。

    .. math::
         w[n] = \alpha - \beta\ \cos \left( \frac{2 \pi n}{N - 1} \right),

    其中， :math:`N` 是全窗口尺寸。

    参数：
        - **periodic** (bool，可选) - 一个标志，表示返回的窗口是否修剪掉来自对称窗口的最后一个重复值。默认值：True。
  
          - 如果为True，则返回的窗口作为周期函数，在上式中， :math:`N = \text{length} + 1` 。
          - 如果为False，则返回一个对称窗口， :math:`N = \text{length}` 。

        - **alpha** (float，可选) - 加权系数，上式中的 :math:`\alpha` ，默认值：0.54。
        - **beta** (float，可选) - 加权系数，上式中的 :math:`\beta` ，默认值：0.46。
        - **dtype** (:class:`mindspore.dtype`，可选) - 数据类型，可选值为 `mindspore.dtype.float16` 、 `mindspore.dtype.float32` 或 `mindspore.dtype.float64` 。默认值： `mindspore.dtype.float32` 。

    输入：
        - **length** (Tensor) - 一个1D的正整数Tensor，控制返回窗口的大小。

    输出：
        Tensor，一个包含窗口的1-D Tensor，其shape为 :math:`\text{length}` 。

    异常：
        - **TypeError** - 如果 `length` 不是一个Tensor。
        - **TypeError** - 如果 `length` 的数据类型不是整型。
        - **TypeError** - 如果 `periodic` 的数据类型不是bool类型。
        - **TypeError** - 如果 `alpha` 的数据类型不是float类型。
        - **TypeError** - 如果 `beta` 的数据类型不是float类型。
        - **TypeError** - 如果 `dtype` 的取值不是 `mindspore.float16` 、 `mindspore.float32` 或 `mindspore.float64` 。
        - **ValueError** - 如果 `length` 的维度不是1。
        - **ValueError** - 如果 `length` 的值是负数。
