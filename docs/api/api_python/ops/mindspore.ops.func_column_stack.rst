mindspore.ops.column_stack
==========================

.. py:function:: mindspore.ops.column_stack(tensors)

    将多个Tensor沿着水平方向维度堆叠成一个Tensor，即按列拼接。Tensor其它维度拼接的结果
    维度不变。类似于 :func:`mindspore.ops.hstack`。

    参数：
        - **tensors** (Union[tuple[Tensor], list[Tensor]]) - 包含多个Tensor。除了需要拼接的轴外，所有的
          Tensors必须有相同的shape。

    返回：
        将输入Tensor堆叠后的 Tensor。

    异常：
        - **TypeError** - 如果 `tensors` 不是 list或tuple。
        - **TypeError** - 如果 `tensors` 的元素不是 Tensor。
        - **ValueError** - 如果 `tensors` 为空。
