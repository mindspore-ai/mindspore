mindspore.ops.gumbel_softmax
============================

.. py:function:: mindspore.ops.gumbel_softmax(logits, tau=1, hard=False, dim=-1)

    返回Gumbel-Softmax分布的Tensor。在 `hard = True` 的时候，返回one-hot形式的离散型Tensor，`hard = False` 时返回在dim维进行过softmax的Tensor。

    参数：
        - **logits** (Tensor) - 输入，是一个非标准化的对数概率分布。只支持float16和float32。
        - **tau** (float) - 标量温度，正数。默认值：1.0。
        - **hard** (bool) - 为True时返回one-hot离散型Tensor，可反向求导。默认值：False。
        - **dim** (int) - 给softmax使用的参数，在dim维上做softmax操作。默认值：-1。

    返回：
        Tensor，shape与dtype和输入 `logits` 相同。

    异常：
        - **TypeError** - `logits` 不是Tensor。
        - **TypeError** - `logits` 不是float16或float32。
        - **TypeError** - `tau` 不是float。
        - **TypeError** - `hard` 不是bool。
        - **TypeError** - `dim` 不是int。
        - **ValueError** - `tau` 不是正数。
