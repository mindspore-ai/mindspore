mindspore.dataset.text.ToVectors
================================

.. py:class:: mindspore.dataset.text.ToVectors(vectors, unk_init=None, lower_case_backup=False)

    根据输入向量表查找向量中的tokens。

    参数：
        - **vectors** (Vectors) - 向量对象。
        - **unk_init** (sequence, 可选) - 用于初始化向量外（OOV）令牌的序列。默认值： ``None`` ，用零向量初始化。
        - **lower_case_backup** (bool, 可选) - 是否查找小写的token。如果为 ``False`` ，则将查找原始大小写中的每个token。
          如果为 ``True`` ，则将首先查找原始大小写中的每个token，如果在属性stoi（字符->索引映射）的键中找不到，则将查找小写中的token。默认值： ``False`` 。

    异常：      
        - **TypeError** - 如果 `unk_init` 不是序列。
        - **TypeError** - 如果 `unk_init` 的元素不是float或int类型。
        - **TypeError** - 如果 `lower_case_backup` 不是bool类型。

    教程样例：
        - `文本变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/text_gallery.html>`_
