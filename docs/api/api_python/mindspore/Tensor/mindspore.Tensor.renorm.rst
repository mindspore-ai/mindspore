mindspore.Tensor.renorm
=======================

.. py:method:: mindspore.Tensor.renorm(p, dim, maxnorm)

    沿维度 `dim` 重新规范Tensor的子张量，并且每个子张量的p范数不超过给定的最大范数 `maxnorm` 。如果子张量的p范数小于 `maxnorm` ，则当前子张量不需要修改；否则该子张量需要修改为对应位置的原值除以该子张量的p范数，然后再乘上 `maxnorm` 。

    参数：
        - **p** (int) - 范数计算的幂。
        - **dim** (int) - 获得子张量的维度。
        - **maxnorm** (float32) - 给定的最大范数。

    返回：
        Tensor，shape和type与输入Tensor一致。

    异常：
        - **TypeError** - `p` 不是int类型。
        - **TypeError** - `dim` 不是int类型。
        - **TypeError** - `maxnorm` 不是float32类型。
        - **ValueError** - `p` 小于等于0。