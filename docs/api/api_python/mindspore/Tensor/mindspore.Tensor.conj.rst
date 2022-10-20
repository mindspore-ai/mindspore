mindspore.Tensor.conj
=====================

.. py:method:: mindspore.Tensor.conj()

    计算输入Tensor的逐元素共轭。复数的形式必须是 `a + bj` ，其中a是实部，b是虚部。

    返回的共轭形式为 `a + bj` 。

    如果 `input` 是实数，则直接返回 `input` 。

    返回：
        Tensor，数据类型与 `input` 相同。

    异常：
        - **TypeError** - 如果当前Tensor的数据类型不是数字类型。
