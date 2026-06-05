mindspore.ops.scatter
=======================================

.. py:function:: mindspore.ops.scatter(input, axis, index, src)

    根据指定索引将 `src` 中的值更新到 `input` 中并返回输出。
    有关更多详细信息，请参阅 :func:`mindspore.ops.tensor_scatter_elements` 。

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：`axis` 将改为 `dim`。

    .. note::
        如果src为tensor，则仅当src的shape和index的shape相同时支持求反向梯度。
        负数 `index` 是非法输入，会导致未定义行为。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (int) - 要进行更新操作的轴。
        - **index** (Tensor) - 要进行更新操作的索引，数据类型为int32或int64。
        - **src** (Tensor, float) - 指定对 `input` 进行更新操作的数据。

    返回：
        Tensor
