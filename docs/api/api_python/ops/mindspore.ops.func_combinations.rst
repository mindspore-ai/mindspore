mindspore.ops.combinations
==========================

.. py:function:: mindspore.ops.combinations(x, r=2, with_replacement=False)

    返回输入Tensor中元素的所有长度为 `r` 的子序列。

    当 `with_replacement` 设为 `False` ，功能与Python里的 `itertools.combinations` 类似，若设为 `True` ，功能与 `itertools.combinations_with_replacement` 一致。

    参数：
        - **x** (Tensor) - 一维Tensor。
        - **r** (int，可选) - 进行组合的元素个数。默认值：2。
        - **with_replacement** (bool，可选) - 是否允许组合存在重复值。默认值：False。

    返回：
        Tensor，包含输入Tensor元素的左右组合值。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
        - **TypeError** - 如果 `r` 不是int类型。
        - **TypeError** - 如果 `with_replacement` 不是bool类型。
        - **ValueError** - 如果 `x` 不是一维。
