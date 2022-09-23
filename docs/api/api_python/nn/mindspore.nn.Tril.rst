mindspore.nn.Tril
=================

.. py:class:: mindspore.nn.Tril

    返回一个Tensor，指定主对角线以上的元素被置为零。

    将矩阵元素沿主对角线分为上三角和下三角（包含对角线）。

    参数 `k` 控制对角线的选择。若 `k` 为0，则沿主对角线分割并保留下三角所有元素。若 `k` 为正值，则沿主对角线向上选择对角线 `k` ，并保留下三角所有元素。若 `k` 为负值，则沿主对角线向下选择对角线 `k` ，并保留下三角所有元素。

    输入：
        - **x** (Tensor)：输入Tensor。数据类型为 `number <https://www.mindspore.cn/docs/zh-CN/r1.9/api_python/mindspore/mindspore.dtype.html#mindspore.dtype>`_ 。
        - **k** (int)：对角线的索引。默认值：0。假设输入的矩阵的维度分别为d1，d2，则k的范围应在[-min(d1, d2)+1, min(d1, d2)-1]，超出该范围时输出值与输入 `x` 一致。

    输出：
        Tensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError：** `k` 不是int。
        - **ValueError：** `x` 的维度小于1。
