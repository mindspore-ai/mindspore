
mindspore.ops.Renorm
=====================

.. py:class:: mindspore.ops.Renorm(p, dim, maxnorm)

    沿维度 `dim` 重新规范输入 `input_x` 的子Tensor，并且每个子Tensor的p范数不超过给定的最大范数 `maxnorm` 。如果子Tensor的p范数小于 `maxnorm` ，则当前子Tensor不需要修改；否则该子Tensor需要修改为对应位置的原值除以该子Tensor的p范数，然后再乘上 `maxnorm` 。

    更多参考详见 :func:`mindspore.ops.renorm` 。