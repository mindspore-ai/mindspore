mindspore.nn.ClipByNorm
========================

.. py:function:: mindspore.nn.ClipByNorm(axis=None)

    对输入Tensor的值进行裁剪，使用 :math:`L_2` 范数控制梯度。

    如果输入Tensor的 :math:`L_2` 范数不大于输入 `clip_norm` ，则此层的输出保持不变。
    否则，Tensor将标准化为：

    .. math::
        \text{output}(X) = \frac{\text{clip_norm} * X}{L_2(X)},

    其中 :math:`L_2(X)` 是 :math:`X` 的 :math:`L_2` 范数。

    **参数：**

    - **axis** (Union[None, int, tuple(int)]) - 指定在哪个维度上计算 :math:`L_2` 范数。如果为None，则计算所有维度。默认值：None。

    **输入：**

    - **x** (Tensor) - 输入n维的Tensor，数据类型为float32或float16。
    - **clip_norm** (Tensor) - shape为 :math:`()` 或 :math:`(1)` 的Tensor。或者Tensor的shape可以广播到输入的shape。

    **输出：**

    Tensor，裁剪后的Tensor与输入 `x` 的shape相同，数据类型为float32。

    **异常：**

    - **TypeError** - `axis` 不是None、int、或tuple。
    - **TypeError** - `x` 的数据类型既不是float16也不是float32。