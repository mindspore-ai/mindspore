mindspore.ops.Einsum
=====================

.. py:class:: mindspore.ops.Einsum(equation)

    此算子使用爱因斯坦求和约定（Einsum）进行Tensor计算。支持对角线、约和、转置、矩阵乘、乘积、内积运算等。

    输入必须是Tensor的tuple。当输入只有一个Tensor时，可以输入(Tensor, )，支持数据类型float16、float32、float64。

    参数：
        - **equation** (str) - 属性，表示要执行的计算。该值只能使用letter([a-z][A-Z])、commas(,)、ellipsis(...)和arrow(->)。letter([a-z][A-Z])表示输入的Tensor的维度，commas(,)表示Tensor维度之间的分隔符，ellipsis(...)表示不关心的Tensor维度，arrow(->)的左侧表示输入Tensor，右侧表示所需的输出维度。

    输入：
        - **x** (Tuple) - 用于计算的输入Tensor，Tensor的数据类型必须相同。

    输出：
        Tensor，shape可以从方程中获得，数据类型与输入Tensor相同。

    异常：
        - **TypeError** - 如果 `equation` 本身无效，或者 `equation` 与输入Tensor不匹配。
