mindspore.ops.Bernoulli
=======================

.. py:class:: mindspore.ops.Bernoulli(seed=-1, offset=0)

    以 `p` 的概率随机将输出的元素设置为0或1，服从伯努利分布。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多细节请参考 :func:`mindspore.ops.bernoulli` 。

    参数：
        - **seed** (int, 可选) - 随机种子，用于生成随机数，数值范围是-1或正整数，-1代表取当前时间戳。默认值： ``-1`` 。
        - **offset** (int, 可选) - 用于在生成随机数序列时改变起始位置。默认值： ``0`` 。

    输入：
        - **x** (Tensor) - Tensor的输入。
        - **p** (Union[Tensor, float], 可选) - 成功概率。 `p` 中每个值代表输出Tensor中对应位置为1的概率，如果是Tensor，其shape必须与 `x` 一致，数值范围在0到1之间。默认值： ``0.5`` 。

    输出：
        - **y** (Tensor) - shape和数据类型与 `x` 相同。
