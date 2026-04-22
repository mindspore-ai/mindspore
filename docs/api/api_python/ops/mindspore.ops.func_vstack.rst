mindspore.ops.vstack
====================

.. py:function:: mindspore.ops.vstack(inputs)

    将多个tensor沿着垂直方向进行堆叠。

    等同于将输入按第一个轴进行拼接。
    一维tensor :math:`(N,)` 重新排列为 :math:`(1, N)` ，然后按第一个轴进行拼接。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：参数 `inputs` 将重命名为 `tensors` 。

    参数：
        - **inputs** (Union(List[tensor], Tuple[tensor])) - 一维或二维输入tensors。

    返回：
        Tensor
