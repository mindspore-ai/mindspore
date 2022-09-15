mindspore.dataset.Dataset.set_dynamic_columns
=============================================

.. py:method:: mindspore.dataset.Dataset.set_dynamic_columns(columns=None)

    设置数据集的动态shape信息，需要在定义好完整的数据处理管道后进行设置。

    参数：
        - **columns** (dict) - 包含数据集中每列shape信息的字典。shape[i]为 `None` 表示shape[i]的数据长度是动态的。
