mindspore.dataset.Dataset.take
===============================

.. py:method:: mindspore.dataset.Dataset.take(count=-1)

    截取数据集的前指定条数据。

    参数：
        - **count** (int, 可选) - 想要截取的数据条数。若该值超过数据集样本总数，则返回全部数据。默认值： ``-1`` ，将返回全部数据。

    .. note::
        当数据处理流水线中存在会改变数据集样本数量的操作时，`take` 操作所处的位置会影响其效果。例如， `batch` 操作会将连续
        指定 `batch_size` 条样本合并成 1 条样本，则 `.batch(batch_size).take(1)` 与 `.take(batch_size).batch(batch_size)` 效果相当。

    返回：
        Dataset，应用了上述操作的新数据集对象。
