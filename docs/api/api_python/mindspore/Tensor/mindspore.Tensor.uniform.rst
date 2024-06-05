mindspore.Tensor.uniform
=======================================

.. py:method:: mindspore.Tensor.uniform(from_=0., to=1., generator=None)

    在半开区间 [from\_, to) 内生成随机数。

    参数：
        - **from_** (number) - 区间的下界。
        - **to** (number) - 区间的上界。
        - **generator** (Generator，可选) - 随机种子。默认值：None。

    返回：
        Tensor，与输入张量形状相同。

    异常：
        - **TypeError** - 如果 `from_` 大于 `to`。
