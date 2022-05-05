mindspore.nn.Jvp
=================

.. py:class:: mindspore.nn.Jvp(fn)

    计算给定网络的雅可比向量积(Jacobian-vector product, JVP)。JVP对应 `前向模式自动微分 <https://www.mindspore.cn/docs/zh-CN/r1.7/design/gradient.html#前向自动微分>`_。

    **参数：**

    - **fn** (Cell) - 基于Cell的网络，用于接收Tensor输入并返回Tensor或者Tensor元组。

    **输入：**

    - **inputs** (Tensor) - 输入网络的入参，单个或多个Tensor。
    - **v** (Tensor or Tuple of Tensor) - 与雅可比矩阵点乘的向量，Shape与网络的输入一致。

    **输出：**

    2个Tensor或Tensor元组构成的元组。

    - **net_output** (Tensor or Tuple of Tensor) - 输入网络的正向计算结果。
    - **jvp** (Tensor or Tuple of Tensor) - 雅可比向量积的结果。
