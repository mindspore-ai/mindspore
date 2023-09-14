mindspore.dataset.Dataset.shuffle
=================================

.. py:method:: mindspore.dataset.Dataset.shuffle(buffer_size)

    通过创建 `buffer_size` 大小的缓存来混洗该数据集。

    1. 生成一个混洗缓冲区包含 `buffer_size` 条数据行。

    2. 从混洗缓冲区中随机选择一个数据行，传递给下一个操作。

    3. 从上一个操作获取下一个数据行（如果有的话），并将其放入混洗缓冲区中。

    4. 重复步骤2和3，直到混洗缓冲区中没有数据行为止。

    在第一个epoch中可以通过 `dataset.config.set_seed` 来设置随机种子。在随后的每个epoch，种子都会被设置成一个新产生的随机值。

    参数：
        - **buffer_size** (int) - 用于混洗的缓冲区大小（必须大于1）。将 `buffer_size` 设置为数据集大小将进行全局混洗。

    返回：
        Dataset，应用了上述操作的新数据集对象。

    异常：
        - **RuntimeError** - 混洗前存在通过 `dataset.sync_wait` 进行同步操作。
