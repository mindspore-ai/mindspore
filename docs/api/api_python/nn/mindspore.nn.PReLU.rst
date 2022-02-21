mindspore.nn.PReLU
===================

.. py:class:: mindspore.nn.PReLU(channel=1, w=0.25)

    PReLU激活层（PReLU Activation Operator）。

    公式定义为：

    .. math::

        prelu(x_i)= \max(0, x_i) + w * \min(0, x_i),

    其中 :math:`x_i` 是输入的Tensor。

    这里 :math:`w` 是一个可学习的参数，默认初始值0.25。
    
    当带参数调用时每个通道上学习一个 :math:`w` 。如果不带参数调用时，则将在所有通道中共享单个参数 :math:`w` 。

    ReLU相关图参见 `PReLU <https://en.wikipedia.org/wiki/Activation_function#/media/File:Activation_prelu.svg>`_ 。

    **参数：**

    - **channel** (int) - 可训练参数 :math:`w` 的数量。它可以是int，值是1或输入Tensor `x` 的通道数。默认值：1。
    - **w** (Union[float, list, Tensor]) - 参数的初始值。它可以是float、float list或与输入Tensor `x` 具有相同数据类型的Tensor。默认值：0.25。

    **输入：**

    - **x** (Tensor) - PReLU的输入，任意维度的Tensor，其数据类型为float16或float32。

    **输出：**

    Tensor，数据类型和shape与 `x` 相同。

    **异常：**

    - **TypeError** - `channel` 不是整数。
    - **TypeError** - `w` 不是float、float list或float Tensor。
    - **TypeError** - `x` 的数据类型既不是float16也不是float32。
    - **ValueError** - `x` 是Ascend上的0-D或1-D Tensor。
    - **ValueError** - `channel` 小于1。