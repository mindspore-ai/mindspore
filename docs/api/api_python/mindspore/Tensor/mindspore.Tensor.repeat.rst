mindspore.Tensor.repeat
=======================

.. py:method:: mindspore.Tensor.repeat(*repeats)

    对张量的每一维度按照指定的重复次数复制其中的元素。

    此方法会拷贝此张量的数据。

    输出张量的shape可以描述如下，其中 :math:`n` 为 `repeats` 中的元素个数：

    .. math::

        shape_{i} = \begin{cases}
        repeats_{i} * input.shape_{i} & \text{if } 0 \le i < input.{rank} \\
        repeats_{i} & \text{if } input.{rank} \le i < n \\
        \end{cases}

    .. note::
        如需对单个轴的每个元素单独指定重复次数，请参考 :func:`mindspore.Tensor.repeat_interleave` 。

    参数：
        - **\*repeats** (int) - `self` 在每个维度上的重复次数，应当为非负数，且 ``1`` 表示在该维度上保持不变。 `repeats` 的元素\
          个数应当不小于 `self` 的维度数。当 `self` 的维度数小于 `repeats` 的元素个数时， `self` 将被广播到与 `repeats` 元素\
          个数相同的维度（如样例所示）。

    返回：
        Tensor，按照指定的重复次数拷贝元素后的新张量。

    异常：
        - **RuntimeError** - `repeats` 的元素个数少于 `self` 的维度数，或 `repeats` 中有元素为负数。
        - **RuntimeError** - `repeats` 的元素个数或 `self` 的维度数超过8。
        - **TypeError** - `repeats` 参数类型错误。

    .. py:method:: mindspore.Tensor.repeat(repeats) -> Tensor
        :noindex:

    对张量的每一维度按照指定的重复次数复制其中的元素。

    此方法会拷贝此张量的数据。

    除了接受变长int型参数，变为接受1个list或tuple型参数外，其他操作与使用 `*repeats` 参数的重载保持一致。

    输出张量的shape可以描述如下，其中 :math:`n` 为 `repeats` 中的元素个数：

    .. math::

        shape_{i} = \begin{cases}
        repeats_{i} * input.shape_{i} & \text{if } 0 \le i < input.{rank} \\
        repeats_{i} & \text{if } input.{rank} \le i < n \\
        \end{cases}

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. note::
        如需对单个轴的每个元素单独指定重复次数，请参考 :func:`mindspore.Tensor.repeat_interleave` 。

    参数：
        - **repeats** (Union[tuple[int], list[int]]) - `self` 在每个维度上的重复次数，应当为非负数，且 ``1`` 表示在该维度上\
          保持不变。 `repeats` 的元素个数应当不小于 `self` 的维度数。当 `self` 的维度数小于 `repeats` 的元素个数时， `self`
          将被广播到与 `repeats` 元素个数相同的维度（如样例所示）。

    返回：
        Tensor，按照指定的重复次数拷贝元素后的新张量。

    异常：
        - **RuntimeError** - `repeats` 的元素个数少于 `self` 的维度数，或 `repeats` 中有元素为负数。
        - **RuntimeError** - `repeats` 的元素个数或 `self` 的维度数超过8。
        - **TypeError** - `repeats` 参数类型错误。

    .. seealso::
        - :func:`mindspore.Tensor.reshape` ：为Tensor指定新的shape，不更改Tensor的数据。
        - :func:`mindspore.Tensor.resize` ：就地改变Tensor的shape和size。
        - :func:`mindspore.Tensor.repeat_interleave` ：将一个Tensor指定轴上的每个元素按指定次数进行重复。
        - :func:`mindspore.Tensor.tile` ：对一个Tensor在每个轴上重复指定次数，但对参数 `repeats` 的个数没有要求。
