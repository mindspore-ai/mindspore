mindspore.dataset.Dataset.output_shapes
=======================================

.. py:method:: mindspore.dataset.Dataset.output_shapes(estimate=False)

    获取数据集对象中每列数据的shape。

    参数：
        - **estimate** (bool) - 如果 `estimate` 为 False，将返回数据集第一条数据的shape。
          否则将遍历整个数据集以获取数据集的真实shape信息，其中动态变化的维度将被标记为None（可用于动态shape数据集场景），默认值：False。

    返回：
        list，每列数据的shape列表。
