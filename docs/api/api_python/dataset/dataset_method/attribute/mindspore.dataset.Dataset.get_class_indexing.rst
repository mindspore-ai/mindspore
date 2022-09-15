mindspore.dataset.Dataset.get_class_indexing
============================================

.. py:method:: mindspore.dataset.Dataset.get_class_indexing()

    返回类别索引。

    返回：
        dict，描述类别名称到索引的键值对映射关系，通常为str-to-int格式。针对COCO数据集，类别名称到索引映射关系描述形式为str-to-list<int>格式，列表中的第二个数字表示超类别。
