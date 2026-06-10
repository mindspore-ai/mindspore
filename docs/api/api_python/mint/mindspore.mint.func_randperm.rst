mindspore.mint.randperm
========================

.. py:function:: mindspore.mint.randperm(n, *, generator=None, dtype=mstype.int64)

    生成从 0 到 n-1 的整数随机排列。

    参数：
        - **n** (Union[Tensor, int]) - 随机排列长度。int或shape为()或(1,)，数据类型为int64的Tensor。
          数值必须大于等于0。当 `n` 为0时，返回空Tensor。

    关键字参数：
        - **generator** (:class:`mindspore.Generator`, 可选) - 伪随机数生成器。默认值： ``None`` ，
          使用默认伪随机数生成器。
        - **dtype** (mindspore.dtype, 可选) - 输出的dtype。默认值：``mstype.int64`` 。

    返回：
        shape为 (`n`,) ，dtype为 `dtype` 的Tensor。

    异常：
        - **TypeError** - 如果 `dtype` 不是一个 `mstype` 类型。
        - **ValueError** - 如果 `n` 是负数。
        - **ValueError** - 如果 `n` 是超过指定数据类型的最大范围。
