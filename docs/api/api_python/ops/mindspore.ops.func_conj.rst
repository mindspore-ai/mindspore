mindspore.ops.conj
===================

.. py:function:: mindspore.ops.conj(input)

    计算输入Tensor的逐元素共轭。复数的形式必须是 `a + bj` ，其中a是实部，b是虚部。

    返回的共轭形式为 `a + bj` 。

    如果 `input` 是实数，则直接返回 `input` 。

    参数：
        - **input** (Tensor) - 输入Tensor，必须是数字类型。

    返回：
        Tensor，数据类型与 `input` 相同。

    异常：
        - **TypeError** -  如果 `input` 的数据类型不是数字类型。
        - **ValueError** - 如果 `input` 不是Tensor。
