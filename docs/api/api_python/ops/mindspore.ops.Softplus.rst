mindspore.ops.Softplus
========================

.. py:class:: mindspore.ops.Softplus

    Softplus激活函数。

    Softplus为ReLU函数的平滑近似。可对一组数值使用来确保转换后输出结果均为正值。函数计算如下：

    .. math::

        \text{output} = \log(1 + \exp(\text{x}))

    输入：
        - **input_x** (Tensor) - shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度，数据类型支持float16或float32。

    输出：
        Tensor，与 `input_x` 的shape和数据类型相同。

    异常：
        - **TypeError** - `input_x` 不是Tensor。
        - **TypeError** - `input_x` 的数据类型非float16或float32。
