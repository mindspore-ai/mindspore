mindspore.ops.flatten
======================

.. py:function:: mindspore.ops.flatten(input_x)

    扁平化（Flatten）输入Tensor，不改变0轴的size。

    参数：
        - **input_x** (Tensor) - 待扁平化的Tensor，其shape为 :math:`(N, \ldots)`， :math:`N` 表示batch size。

    返回：
        Tensor，其shape为 :math:`(N, X)` 的Tensor，其中 :math:`X` 是余下维度的乘积。

    异常：
        - **TypeError** - `input_x` 不是Tensor。
        - **ValueError** - `input_x` 的shape长度小于1。