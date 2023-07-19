mindspore.ops.Dense
===================

.. py:class:: mindspore.ops.Dense(has_bias=True)

    全连接融合算子。

    适用于输入的密集连接算子。算子的实现如下：

    .. math::
        output = x * w + b,

    其中  :math:`x` 是输入Tensor， :math:`w` 是一个权重矩阵，其数据类型与 :math:`x` 相同， :math:`b` 是一个偏置向量，其数据类型与 :math:`b` 相同（仅当 `has_bias` 为 ``True`` 时）。

    参数：
        - **has_bias** (bool，可选) - 是否使用偏置向量 :math:`b` 。默认值： ``True`` 。

    输入：
        - **x** (Union[Tensor, Parameter]) - 输入Tensor，其数据类型为float16、float32或float64。shape必须满足： :math:`len(x.shape)>1` 。
        - **w** (Union[Tensor, Parameter]) - 权重Tensor，其数据类型为float16、float32或float64。shape必须满足： :math:`len(w.shape)=2` ， :math:`w.shape[0]=x.shape[-2]` ， :math:`w.shape[1]=x.shape[-1]` 。
        - **b** (Union[Tensor, Parameter]) - 偏置Tensor，其数据类型为float16、float32或float64。当 `has_bias` 为 ``True`` 时，shape必须满足： :math:`len(b.shape)=1`， :math:`b.shape[0]=x.shape[-2]` 。

    输出：
        shape为 :math:`(*x.shape[:-1], w.shape[0])` 的Tensor。

    异常：
        - **TypeError** - `has_bias` 不是bool值。
