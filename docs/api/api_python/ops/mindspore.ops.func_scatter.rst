mindspore.ops.scatter
=======================================

.. py:function:: mindspore.ops.scatter(input, axis, index, src)

    根据指定索引将 `src` 中的值更新到 `input` 中返回输出。
    有关更多详细信息，请参阅 :func:`mindspore.ops.tensor_scatter_elements` 。

    参数：
        - **input** (Tensor) - 输入Tensor。 `input` 的秩必须至少为1。
        - **axis** (int) - 要进行更新操作的轴。取值范围是[-r, r)，其中r是 `input` 的秩。
        - **index** (Tensor) - 输入Tensor的索引，数据类型为int32或int64的。其rank必须和 `input` 一致。取值范围是[-s, s)，这里的s是 `input` 在 `axis` 指定轴的size。
        - **src** (Tensor) - 指定与 `input` 进行更新操作的Tensor，其数据类型与输入 `input` 的数据类型相同。 `src` 的shape必须等于 `index` 的shape。

    返回：
        Tensor，shape和数据类型与输入 `input` 相同。

    异常：
        - **TypeError** - `index` 的数据类型既不是int32，也不是int64。
        - **ValueError** - `input` 、 `index` 和 `src` 中，任意一者的秩小于1。
        - **ValueError** - `src` 的shape和 `index` 的shape不一样。
        - **ValueError** - `src` 的秩和 `input` 的秩不一样。
        - **RuntimeError** - `input` 的数据类型和 `src` 的数据类型不能隐式转换。