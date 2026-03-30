mindspore.ops.gumbel_softmax
============================

.. py:function:: mindspore.ops.gumbel_softmax(logits, tau=1.0, hard=False, dim=-1)

    返回Gumbel-Softmax分布的Tensor。当 `hard` 为 ``True`` 时，返回one-hot形式的离散型Tensor；当 `hard` 为 ``False`` 时，返回在 `dim` 维度上进行softmax运算后的Tensor。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将新增可选参数 `eps` ，接口签名将变更为
        ``mindspore.ops.gumbel_softmax(logits, tau=1.0, hard=False, eps=1e-10, dim=-1)`` 。
        该变更与当前版本兼容，不使用 `eps` 参数时行为不变。

    参数：
        - **logits** (Tensor) - 输入，是一个非标准化的对数概率分布。只支持float16和float32。
        - **tau** (float，可选) - 标量温度，正数。默认 ``1.0`` 。
        - **hard** (bool，可选) - 为 ``True`` 时返回one-hot离散型Tensor，可反向求导。默认 ``False`` 。
        - **dim** (int，可选) - 给softmax使用的参数，在dim维上做softmax操作。默认 ``-1`` 。

    返回：
        Tensor，shape与dtype和输入 `logits` 相同。

    异常：
        - **TypeError** - `logits` 不是Tensor。
        - **TypeError** - `logits` 不是float16或float32。
        - **TypeError** - `tau` 不是float。
        - **TypeError** - `hard` 不是bool。
        - **TypeError** - `dim` 不是int。
        - **ValueError** - `tau` 不是正数。
