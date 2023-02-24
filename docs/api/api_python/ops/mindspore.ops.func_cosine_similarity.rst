mindspore.ops.cosine_similarity
================================

.. py:function:: mindspore.ops.cosine_similarity(x1, x2, dim=1, eps=1e-08)

    沿轴计算的x1和x2之间的余弦相似度。

    .. note::
        当前暂不支持对输入进行广播。

    .. math::
        \text{similarity} = \dfrac{x_1 \cdot x_2}{\max(\Vert x_1 \Vert _2 \cdot \Vert x_2 \Vert _2, \epsilon)}

    参数：
        - **x1** (Tensor) - 第一个输入Tensor。
        - **x2** (Tensor) - 第二个输入Tensor。
        - **dim** (int, optional) - 计算余弦相似度的轴。默认值：1。
        - **eps** (float, optional) - 极小值，用于避免除零的情况。默认值：1e-8。

    返回：
        Tensor，x1和x2之间的余弦相似度。

    异常：
        - **TypeError** - 如果 `x1` 或 `x2` 的数据类型既不是float16也不是float32。