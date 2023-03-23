mindspore.ops.dstack
====================

.. py:function:: mindspore.ops.dstack(inputs)

    将多个Tensor沿着第三维度进行堆叠。

    1-D Tensor :math:`(N,)` 重新排列为 :math:`(1,N,1)` ，2-D Tensor :math:`(M,N)` 重新排列为 :math:`(M,N,1)` 。

    参数：
        - **inputs** (Union(List[Tensor], Tuple[Tensor])) - 一个Tensor序列。除了第三个轴外，所有的
          Tensor必须有相同的shape。如果是1-D或2-D的Tensor，则它们的shape必须相同。

    返回：
        堆叠后的Tensor，其维度至少为3。输出shape与 `numpy.dstack()` 类似。

    异常：
        - **TypeError** - 如果 `inputs` 不是list或tuple。
        - **ValueError** - 如果 `inputs` 为空。
