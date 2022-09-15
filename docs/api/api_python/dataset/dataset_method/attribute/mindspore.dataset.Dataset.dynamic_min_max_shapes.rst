mindspore.dataset.Dataset.dynamic_min_max_shapes
================================================

.. py:method:: mindspore.dataset.Dataset.dynamic_min_max_shapes()

    当数据集对象中的数据shape不唯一（动态shape）时，获取数据的最小shape和最大shape。

    返回：
        两个列表代表最小shape和最大shape，每个列表中的shape按照数据列的顺序排列。
