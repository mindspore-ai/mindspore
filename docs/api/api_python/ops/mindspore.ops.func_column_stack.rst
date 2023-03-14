mindspore.ops.column_stack
==========================

.. py:function:: mindspore.ops.column_stack(input)

    将多个1-D 或2-D Tensor沿着水平方向堆叠成一个2-D Tensor，即按列拼接。2-D Tensor拼接的结果
    仍为2-D Tensor。类似于 :func:`mindspore.ops.hstack`。

    参数：
        - **input** (Union[Tensor, tuple, list]) - 多个1-D 或 2-D的Tensor。除了需要拼接的轴外，所有的
          Tensors必须有相同的shape。

    返回：
        将输入Tensor堆叠后的2-D Tensor。

    异常：
        - **TypeError** - 如果 `input` 不是 Tensor、list或tuple。
        - **ValueError** - 如果 `input` 为空。
