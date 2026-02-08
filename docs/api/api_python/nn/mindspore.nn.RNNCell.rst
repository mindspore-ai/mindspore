mindspore.nn.RNNCell
=====================

.. py:class:: mindspore.nn.RNNCell(input_size, hidden_size, has_bias=True, nonlinearity='tanh', dtype=mstype.float32)

    循环神经网络单元，激活函数是tanh或relu。

    .. math::
        h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})

    其中 :math:`h_t` 是在 `t` 时刻的隐藏状态， :math:`x_t` 是在 `t` 时刻的输入， :math:`h_{(t-1)}` 是在 :math:`t-1` 时刻的隐藏状态，或初始隐藏状态。

    参数：
        - **input_size** (int) - 输入层输入的特征向量维度。
        - **hidden_size** (int) - 隐藏层输出的特征向量维度。
        - **has_bias** (bool) - Cell是否有偏置项 :math:`b_{ih}` 和 :math:`b_{hh}` 。默认值： ``True`` 。
        - **nonlinearity** (str) - 用于选择非线性激活函数。取值可以是'tanh'或'relu'。默认值： ``'tanh'`` 。
        - **dtype** (:class:`mindspore.dtype`) - Parameters的dtype。默认值： ``mstype.float32`` 。

    输入：
        - **x** (Tensor) - 输入Tensor，其shape为 :math:`(batch\_size, input\_size)` 。
        - **hx** (Tensor) - 输入Tensor，其数据类型为mindspore.float32及shape为 :math:`(batch\_size, hidden\_size)` 。

    输出：
        - **hx'** (Tensor) - shape为 :math:`(batch\_size, hidden\_size)` 的Tensor。

    异常：
        - **TypeError** - `input_size` 或 `hidden_size` 不是int或不大于0。
        - **TypeError** - `has_bias` 不是bool。
        - **ValueError** - `nonlinearity` 不在['tanh', 'relu']中。