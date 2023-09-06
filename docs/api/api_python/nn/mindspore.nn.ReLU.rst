mindspore.nn.ReLU
=================

.. py:class:: mindspore.nn.ReLU

    修正线性单元激活函数（Rectified Linear Unit activation function）。

    .. math::

        \text{ReLU}(x) = (x)^+ = \max(0, x),

    逐元素求 :math:`\max(0, x)` 。
    
    .. note::
        负数输出值会被修改为0，正数输出不受影响。

    ReLU激活函数图：

    .. image:: images/ReLU.png
        :align: center

    输入：
        - **x** (Tensor) - 用于计算ReLU的任意维度的Tensor。数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_。

    输出：
        Tensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 的数据类型不是number。
