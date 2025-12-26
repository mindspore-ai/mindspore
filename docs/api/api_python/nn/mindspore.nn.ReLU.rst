mindspore.nn.ReLU
=================

.. py:class:: mindspore.nn.ReLU

    逐元素计算ReLU（修正线性单元）激活函数。

    .. math::

        \text{ReLU}(input) = (input)^+ = \max(0, input),

    逐元素求 :math:`\max(0, input)` 。

    .. note::
        输出为负的神经元会被抑制，正值保持不变。

    ReLU激活函数图：

    .. image:: ../images/ReLU.png
        :align: center

    输入：
        - **input** (Tensor) - 输入 tensor。

    输出：
        Tensor。
