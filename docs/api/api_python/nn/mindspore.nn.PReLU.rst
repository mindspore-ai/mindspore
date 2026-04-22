mindspore.nn.PReLU
===================

.. py:class:: mindspore.nn.PReLU(channel=1, w=0.25)

    逐元素计算PReLU（PReLU Activation Operator）激活函数。

    .. warning::
        从2.9.0（不含）之后版本开始，参数 `channel` 和 `w` 将分别重命名为 `num_parameters` 和 `init`。

    公式定义为：

    .. math::

        PReLU(x_i)= \max(0, x_i) + w * \min(0, x_i),

    其中 :math:`x_i` 是输入的Tensor。

    这里 :math:`w` 是一个可学习的参数，默认初始值 ``0.25``。

    当带参数调用时每个通道上学习一个 :math:`w` 。如果不带参数调用时，则将在所有通道中共享单个参数 :math:`w` 。

    PReLU函数图：

    .. image:: ../images/PReLU.png
        :align: center

    参数：
        - **channel** (int，可选) - 可训练参数 :math:`w` 的数量。它可以是int，值是 ``1``，或输入Tensor `x` 的通道数。默认值： ``1`` 。
        - **w** (Union[float, list, Tensor]，可选) - 参数的初始值。它可以是float，float组成的list，或与输入Tensor `x` 具有相同数据类型的Tensor。默认值： ``0.25`` 。

    输入：
        - **x** (Tensor) - PReLU的输入Tensor，其shape为 :math:`(N, *)` ，其中 :math:`*` 表示任意的额外维度，数据类型为float16或float32。

    输出：
        Tensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `channel` 不是int。
        - **TypeError** - `w` 既不是float，也不是list[float]或Tensor[float]。
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。
        - **ValueError** - `x` 是Ascend上的0-D或1-D Tensor。
        - **ValueError** - `channel` 小于1。
