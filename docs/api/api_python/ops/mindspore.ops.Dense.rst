mindspore.ops.Dense
===================

.. py:class:: mindspore.ops.Dense()

    全连接融合算子。

    适用于输入的密集连接算子。算子的实现如下：

    .. math::
        output = x @ w ^ T + b,

    其中  :math:`x` 是输入Tensor， :math:`w` 是一个权重矩阵，其数据类型与 :math:`x` 相同， :math:`b` 是一个偏置向量，其数据类型与 :math:`x` 相同（仅当 `b` 不为 ``None`` 时）。

    输入：
        - **x** (Tensor) - 输入Tensor。shape必须满足： :math:`len(x.shape)>0` 。
        - **w** (Tensor) - 权重Tensor。shape必须满足： 若 :math:`len(x.shape)>1` ，则 :math:`len(w.shape)=2` 。若 :math:`len(x.shape)=1` ，则 :math:`len(w.shape)=1` 。 :math:`w.shape[-1]=x.shape[-1]` 。
        - **b** (Union[Tensor, None]) - 偏置Tensor。当 `b` 不为 ``None`` 时，shape必须满足： 若 :math:`len(x.shape)>1` ，则 :math:`len(b.shape)=0` 或 :math:`len(b.shape)=1` 。若 :math:`len(b.shape)=1` ， 则 :math:`b.shape[0]=w.shape[0]` 。若 :math:`len(x.shape)=1` ，则 :math:`len(b.shape)=0` 。

    输出：
        若 :math:`len(x.shape)>1` ，则输出shape为 :math:`(*x.shape[:-1], w.shape[0])` 的Tensor。若 :math:`len(x.shape)=1` ，则输出shape为 :math:`()` 的Tensor。
