mindspore.ops.vstack
====================

.. py:function:: mindspore.ops.vstack(inputs)

    将多个Tensor沿着竖直方向进行堆叠。

    相当于将输入沿着第一个轴进行拼接。
    1-D Tensor :math:`(N,)` 重新排列为 :math:`(1, N)` ，然后沿着第一个轴进行拼接。

    参数：
        - **inputs** (Union(List[tensor], Tuple[tensor])) - 一个1-D或2-D Tensor序列。除了第一个轴外，所有的
          Tensor必须有相同的shape。如果是1-DTensor，则它们的shape必须相同。

    返回：
        堆叠后的Tensor，其维度至少为3。输出shape与 `numpy.vstack()` 类似。

    异常：
        - **TypeError** - 如果 `inputs` 不是list或tuple。
        - **ValueError** - 如果 `inputs` 为空。
