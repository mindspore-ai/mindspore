mindspore.ops.NonDeterministicInts
===================================

.. py:class:: mindspore.ops.NonDeterministicInts(dtype=mstype.int64)

    生成指定数据类型范围内的随机整数。

    返回Tensor的shape由输入 `shape` 指定，其中的随机数从指定数据类型的可表示范围内抽取。

    .. warning::
        `shape` 中的值必须大于零，输出元素总数不可超过1000000。

    参数：
        - **dtype** (mindspore.dtype，可选) - 输出数据类型。支持的数据类型为： ``mstype.int32`` 和 ``mstype.int64`` 。默认值： ``mstype.int64`` 。

    输入：
        - **shape** (Tensor) - 输出Tensor的shape。支持的数据类型为：int32和int64。

    输出：
        Tensor，其shape由输入 `shape` 指定，数据类型由参数 `dtype` 指定。

    异常：
        - **TypeError** - `shape` 不是Tensor。
        - **TypeError** - `dtype` 不是mstype.int32或mstype.int64。
        - **ValueError** - `shape` 中含有负数。
        - **ValueError** - `shape` 元素个数少于2。
        - **ValueError** - `shape` 不是一维Tensor。
        - **ValueError** - 输出元素总个数大于1000000。
