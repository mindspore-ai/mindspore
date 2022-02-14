mindspore.dataset.transforms.c_transforms.TypeCast
==================================================

.. py:class:: mindspore.dataset.transforms.c_transforms.TypeCast(data_type)

    将输入的Tensor转换为指定的数据类型。

    **参数：**

    - **data_type** (mindspore.dtype) - 指定要转换的数据类型。

    **异常：**
      
    - **TypeError** - 参数 `data_type` 类型不为 MindSpore支持的数据类型，如bool、int、float 或 string。
