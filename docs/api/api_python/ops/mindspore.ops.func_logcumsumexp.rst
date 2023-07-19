mindspore.ops.logcumsumexp
==========================

.. py:function:: mindspore.ops.logcumsumexp(input, axis)

    计算输入Tensor `input` 元素的的指数沿轴 `axis` 的累积和的对数。例如，如果 `input` 是 tensor [a, b, c] 并且 `input` 是0，返回值为 [a, log(exp(a) + exp(b)),
    log(exp(a) + exp(b) + exp(c))]。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 任意维度的Tensor。必须是以下几种数据类型：float16、float32、float64。
        - **axis** (int) - 累积计算的轴。必须在 [-rank(x), rank(x)) 的范围之内. 

    返回：
        Tensor，和输入Tensor的dtype和shape相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `input` 的dtype不在 [float16, float32, float64] 之内。
        - **TypeError** - 如果 `axis` 不是int。
        - **ValueError** - 如果 `axis` 超出范围 [-rank(input), rank(input)) 。
