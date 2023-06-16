mindspore.ops.CeLU
===================

.. py:class:: mindspore.ops.CeLU(alpha=1.0)

    逐元素计算输入Tensor的CeLU（连续可微指数线性单位）。

    更多参考详见 :func:`mindspore.ops.celu`。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **alpha** (float，可选) - celu公式定义的阈值 :math:`\alpha` 。默认值： ``1.0`` 。

    输入：
        - **input_x** (Tensor) - 输入Tensor，数据类型为float16或float32。

    输出：
        Tensor，shape和数据类型与输入相同。
       