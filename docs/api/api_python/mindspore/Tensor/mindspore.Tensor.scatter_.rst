mindspore.Tensor.scatter\_
==========================

.. py:method:: mindspore.Tensor.scatter_(dim, index, src) -> Tensor

    根据 `index` 使用 `src` 中的值更新当前张量 `self` 。

    对当前张量 `self` 被 `dim` 选中的维度使用 `index` 进行索引，对其他维度按顺序遍历，将 `src` 中的值更新到 `self` 中，并返回 `self` 自身。
    此操作是 :func:`mindspore.Tensor.gather` 的原地更新版本的逆操作。

    此操作提供另外3个重载，对 `reduce` 参数和标量值的支持。

    下面看一个三维的例子：

    .. code-block::

        self[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0

        self[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

    .. warning::
        - 如果 `index` 有多个索引指向 `self` 内的同一个位置，则 `self` 中该位置的最终值是不确定的。
        - 在Ascend平台上，如果 `index` 中元素的值不在 `[-self.shape[dim], self.shape[dim])` 的范围内，则其行为是不确定的。
        - 这是一个实验性API，后续可能修改或删除。

    .. note::
        仅当 `src` 的shape和 `index` 的shape相同时，支持求 `self` 向 `src` 的反向梯度。

    参数：
        - **dim** (int) - 要进行操作的轴。取值范围是 `[-r, r)` ，其中 `r` 是 `self` 的秩。
        - **index** (Tensor) - 在 `dim` 指定的目标轴上访问 `self` 时使用的索引，数据类型为int32或int64。如果为空Tensor，则将直接返回，不进行任何操作；否则其rank必须和 `self` 一致，且每个元素取值范围是 `[-s, s)` ，这里的 `s` 是 `self` 在 `dim` 指定轴的大小。
        - **src** (Tensor) - 指定对 `self` 进行更新操作的数据。其rank与dtype必须与 `self` 的相同。

    返回：
        Tensor，返回被修改后的 `self` 自身。

    异常：
        - **TypeError** - `self` 、 `index` 或 `src` 的类型不支持。
        - **RuntimeError** - `dim` 的取值超出 `[-r, r)` 的限制。
        - **RuntimeError** - `self` 的秩超过8。
        - **RuntimeError** - 向量 `self` ， `index` 或 `src` 的dtype不被支持。
        - **RuntimeError** - `self` 与 `src` 类型不一致。
        - **RuntimeError** - `self` 、 `index` 与 `src` 的秩不一致且 `index` 不为空。
        - **RuntimeError** - 存在一个维度 `d` 使得 `index.size(d) > src.size(d)`。
        - **RuntimeError** - 存在一个维度 `d` 使得 `index.size(d) > self.size(d)`。

    .. py:method:: mindspore.Tensor.scatter_(dim, index, src, *, reduce) -> Tensor
        :noindex:

    根据 `index` 使用 `src` 中的值更新当前张量 `self` 。

    使用 `reduce` 指定的归约操作，对 `self` 被 `dim` 选中的维度使用 `index` 进行索引，对其他维度按顺序遍历，将 `src` 中的值累加或累乘到 `self` 中，并返回 `self` 自身。
    此操作是 :func:`mindspore.Tensor.gather` 的原地更新版本的逆操作。
    除替换操作将根据参数 `reduce` 的值更改为累加或累乘外，其他操作与不带有 `reduce` 参数的接受 `src` 的重载保持一致。

    下面看一个三维的例子：

    .. code-block::

        self[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0, reduce == "add"

        self[i][j][index[i][j][k]] *= src[i][j][k]  # if dim == 2, reduce == "multiply"

    .. warning::
        - 如果 `index` 有多个索引指向 `self` 内的同一个位置，则 `self` 中该位置的最终值是不确定的。
        - 在Ascend平台上，如果 `index` 中元素的值不在 `[-self.shape[dim], self.shape[dim])` 的范围内，则其行为是不确定的。
        - 这是一个实验性API，后续可能修改或删除。

    .. note::
        此重载不支持反向梯度计算，如计算梯度将返回全0结果。

    参数：
        - **dim** (int) - 要进行操作的轴。取值范围是 `[-r, r)` ，其中 `r` 是 `self` 的秩。
        - **index** (Tensor) - 在 `dim` 指定的目标轴上访问 `self` 时使用的索引，数据类型为int32或int64。如果为空Tensor，则将直接返回，不进行任何操作；否则其rank必须和 `self` 一致，且每个元素取值范围是 `[-s, s)` ，这里的 `s` 是 `self` 在 `dim` 指定轴的大小。
        - **src** (Tensor) - 指定对 `self` 进行更新操作的数据。其rank与dtype必须与 `self` 的相同。

    关键字参数：
        - **reduce** (str) - 进行的归约操作，支持 ``"add"`` ， ``"multiply"`` 。当 `reduce` 设置为 ``"add"`` 时，`src` 将根据 `index` 累加到 `self` 。当 `reduce` 设置为 ``"multiply"`` 时，`src` 将根据 `index` 累乘到 `self` 。

    返回：
        Tensor，返回被修改后的 `self` 自身。

    异常：
        - **TypeError** - `self` 、 `index` 或 `src` 的类型不支持。
        - **ValueError** - `reduce` 为字符串但取值不为 ``"add"`` ， ``"multiply"`` 。
        - **RuntimeError** - `dim` 的取值超出 `[-r, r)` 的限制。
        - **RuntimeError** - `self` 的秩超过8。
        - **RuntimeError** - 向量 `self` ， `index` 或 `src` 的dtype不被支持。
        - **RuntimeError** - `self` 与 `src` 类型不一致。
        - **RuntimeError** - `self` 、 `index` 与 `src` 的秩不一致且 `index` 不为空。
        - **RuntimeError** - 存在一个维度 `d` 使得 `index.size(d) > src.size(d)`。
        - **RuntimeError** - 存在一个维度 `d` 使得 `index.size(d) > self.size(d)`。

    .. py:method:: mindspore.Tensor.scatter_(dim, index, value) -> Tensor
        :noindex:

    根据 `index` 使用 `value` 更新当前张量 `self` 。

    对 `self` 被 `dim` 选中的维度使用 `index` 进行索引，对其他维度按顺序遍历，将 `value` 中的值更新到 `self` 中，并返回 `self` 自身。
    此操作是 :func:`mindspore.Tensor.gather` 的原地更新版本的逆操作。
    可以认为将 `value` 广播为shape及dtype与 `self` 一致的Tensor后，其他操作与不带有 `reduce` 参数的接受 `src` 的重载保持一致。

    下面看一个三维的例子：

    .. code-block::

        self[index[i][j][k]][j][k] = value  # if dim == 0

        self[i][j][index[i][j][k]] = value  # if dim == 2

    .. warning::
        - 如果 `index` 有多个索引指向 `self` 内的同一个位置，则 `self` 中该位置的最终值是不确定的。
        - 在Ascend平台上，如果 `index` 中元素的值不在 `[-self.shape[dim], self.shape[dim])` 的范围内，则其行为是不确定的。
        - 这是一个实验性API，后续可能修改或删除。

    参数：
        - **dim** (int) - 要进行操作的轴。取值范围是 `[-r, r)` ，其中 `r` 是 `self` 的秩。
        - **index** (Tensor) - 在 `dim` 指定的目标轴上访问 `self` 时使用的索引，数据类型为int32或int64。如果为空Tensor，则将直接返回，不进行任何操作；否则其rank必须和 `self` 一致，且每个元素取值范围是 `[-s, s)` ，这里的 `s` 是 `self` 在 `dim` 指定轴的大小。
        - **value** (int, float, bool) - 指定对 `self` 进行更新操作的数据。可视为将尝试将其广播为shape及dtype与 `self` 一致的Tensor并视为 `src` 参与运算。

    返回：
        Tensor，返回被修改后的 `self` 自身。

    异常：
        - **TypeError** - `self` 、 `index` 或 `value` 的类型不支持。
        - **RuntimeError** - `dim` 的取值超出 `[-r, r)` 的限制。
        - **RuntimeError** - `self` 的秩超过8。
        - **RuntimeError** - 张量 `self` 或 `index` 的dtype不被支持。
        - **RuntimeError** - `index` 不为空且秩与 `self` 不一致。
        - **RuntimeError** - 存在一个维度 `d` 使得 `index.size(d) > self.size(d)`。

    .. py:method:: mindspore.Tensor.scatter_(dim, index, value, *, reduce) -> Tensor
        :noindex:

    根据 `index` 使用 `value` 更新当前张量 `self` 。

    使用 `reduce` 指定的归约操作，对 `self` 被 `dim` 选中的维度使用 `index` 进行索引，对其他维度按顺序遍历，将 `value` 中的值累加或累乘到 `self` 中，并返回 `self` 自身。
    此操作是 :func:`mindspore.Tensor.gather` 的原地更新版本的逆操作。
    除替换操作将根据参数 `reduce` 的值更改为累加或累乘外，其他行为与不带有 `reduce` 参数的接受 `value` 的重载保持一致。

    下面看一个三维的例子：

    .. code-block::

        self[i][index[i][j][k]][k] += value  # if dim == 1, reduce == "add"

        self[i][j][index[i][j][k]] *= value  # if dim == 2, reduce == "multiply"

    .. warning::
        - 如果 `index` 有多个索引指向 `self` 内的同一个位置，则 `self` 中该位置的最终值是不确定的。
        - 在Ascend平台上，如果 `index` 中元素的值不在 `[-self.shape[dim], self.shape[dim])` 的范围内，则其行为是不确定的。
        - 这是一个实验性API，后续可能修改或删除。

    .. note::
        此重载不支持反向梯度计算，如计算梯度将返回全0结果。

    参数：
        - **dim** (int) - 要进行操作的轴。取值范围是 `[-r, r)` ，其中 `r` 是 `self` 的秩。
        - **index** (Tensor) - 在 `dim` 指定的目标轴上访问 `self` 时使用的索引，数据类型为int32或int64。如果为空Tensor，则将直接返回，不进行任何操作；否则其rank必须和 `self` 一致，且每个元素取值范围是 `[-s, s)` ，这里的 `s` 是 `self` 在 `dim` 指定轴的大小。
        - **value** (int, float, bool) - 指定对 `self` 进行更新操作的数据。可视为将尝试将其广播为shape及dtype与 `self` 一致的Tensor并视为 `src` 参与运算。

    关键字参数：
        - **reduce** (str) - 进行的归约操作，支持 ``"add"`` ， ``"multiply"`` 。当 `reduce` 设置为 ``"add"`` 时， `src` 将根据 `index` 累加到 `self` 。当 `reduce` 设置为 ``"multiply"`` 时， `src` 将根据 `index` 累乘到 `self` 。

    返回：
        Tensor，返回被修改后的 `self` 自身。

    异常：
        - **TypeError** - `self` 、 `index` 或 `value` 的类型不支持。
        - **ValueError** - `reduce` 为字符串但取值不为 ``"add"`` ， ``"multiply"`` 。
        - **RuntimeError** - `dim` 的取值超出 `[-r, r)` 的限制。
        - **RuntimeError** - `self` 的秩超过8。
        - **RuntimeError** - 张量 `self` 或 `index` 的dtype不被支持。
        - **RuntimeError** - `index` 不为空且秩与 `self` 不一致。
        - **RuntimeError** - 存在一个维度 `d` 使得 `index.size(d) > self.size(d)`。
