mindspore.dataset.sync_wait_for_dataset
=======================================

.. py:function:: mindspore.dataset.sync_wait_for_dataset(rank_id, rank_size, current_epoch)

    等待所有的卡需要的数据集文件下载完成。

    .. note:: 需要配合 `mindspore.dataset.OBSMindDataset` 使用，建议在每次epoch开始前调用。

    参数：
        - **rank_id** (int) - 当前卡的逻辑序号。
        - **rank_size** (int) - 卡的数量。
        - **current_epoch** (int) - 训练时当前的epoch数。
