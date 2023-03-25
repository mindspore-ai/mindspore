
mindspore.ops.Renorm
=====================

.. py:class:: mindspore.ops.Renorm(p, dim, maxnorm)

    对Tensor沿着指定维度 `dim` 进行重新规范化，要求每个子Tensor的 `p` 范数不超过 `maxnorm` 。如果子Tensor的 `p` 范数小于 `maxnorm` ，则其值不需要改变。否则，子Tensor需要修改为相应位置的原始值除以子Tensor的p范数，然后再乘以 `maxnorm` 。

    更多参考详见 :func:`mindspore.ops.renorm` 。
