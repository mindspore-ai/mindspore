mindspore.ops.renorm
====================

.. py:function:: mindspore.ops.renorm(input, p, axis, maxnorm)

    对Tensor沿着指定维度 `dim` 进行重新规范化，要求每个子Tensor的 `p` 范数不超过 `maxnorm` 。如果子Tensor的 `p` 范数小于 `maxnorm` ，则其值不需要改变。否则，子Tensor需要修改为相应位置的原始值除以子Tensor的p范数，然后再乘以 `maxnorm` 。

    参数：
        - **input** (Tensor) - 输入张量，类型为float32或者float16。
        - **p** (int) - 范数计算的幂。
        - **axis** (int) - 获得子张量的维度。
        - **maxnorm** (float32) - 给定的最大范数。

    返回：
        Tensor，shape和type与输入Tensor一致。

    异常：
        - **TypeError** - `p` 不是int类型。
        - **TypeError** - `axis` 不是int类型。
        - **TypeError** - `maxnorm` 不是float32类型。
        - **ValueError** - `p` 小于1。
