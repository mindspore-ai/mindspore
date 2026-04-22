mindspore.ops.relu6
====================

.. py:function:: mindspore.ops.relu6(x)

    计算输入Tensor的ReLU（修正线性单元），其上限为6。

    .. math::
        \text{ReLU6}(x) = \min(\max(0,x), 6)

    返回 :math:`\min(\max(0,x), 6)` 元素的值。

    ReLU6函数图：

    .. image:: ../images/ReLU6.png
        :align: center

    .. warning::
        从2.9.0（不含）之后版本开始，参数 `x` 将重命名为 `input`，并新增 `inplace` 参数。

    参数：
        - **x** (Tensor) - 输入Tensor。shape为 :math:`(N, *)` ，其中 :math:`*` 表示任意的附加维度数。数据类型必须为float16、float32。

    返回：
        Tensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `x` 的数据类型不是float16或float32。
        - **TypeError** - `x` 不是Tensor。
