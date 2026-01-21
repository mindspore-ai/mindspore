mindspore.ops.diff
==================

.. py:function:: mindspore.ops.diff(x, n=1, axis=-1, prepend=None, append=None)

    按指定轴计算输入tensor的n阶前向差分。

    一阶差分的计算公式为：:math:`out[i] = x[i+1] - x[i]` 。
    高阶差分通过递归调用 :func:`mindspore.ops.diff` 实现。

    .. note::
        不支持空Tensor，如果传入了空Tensor，会出现ValueError。空Tensor指的是，Tensor的任意一维为零。比如shape为 :math:`(0,)`  ， :math:`(1, 2, 0, 4)` 的Tensor都为空Tensor。

    参数：
        - **x** (Tensor) - 输入tensor。
        - **n** (int，可选) - 计算差分的阶数，目前只支持 ``1``。默认 ``1`` 。
        - **axis** (int，可选) - 指定轴。默认 ``-1`` 。
        - **prepend** (Tensor，可选) - 在计算差分之前，沿 `axis` 添加到 `x` 之前的值。它们的维度必须与输入的维度相同，除维度 `axis` 外，其余维度的形状必须与输入张量一致。默认 ``None`` 。
        - **append** (Tensor，可选) - 在计算差分之前，沿 `axis` 添加到 `x` 之后的值。它们的维度必须与输入的维度相同，除维度 `axis` 外，其余维度的形状必须与输入张量一致。默认 ``None`` 。

    返回：
        一个数据类型与 `x` 相同，shape与 `x` 相比在第 `axis` 维上缩小 `n` 的tensor。
