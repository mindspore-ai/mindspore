mindspore.dataset.transforms.c_transforms.Fill
==============================================

.. py:class:: mindspore.dataset.transforms.c_transforms.Fill(fill_value)

    将Tensor的所有元素都赋值为指定的值，输出Tensor将具有与输入Tensor具有相同的shape和数据类型。

    **参数：**

    - **fill_value** (Union[str, bytes, int, float, bool]) - 用于填充Tensor的值。

    **异常：**
      
    - **TypeError** - 参数 `fill_value` 类型不为 string、float、bool、int 或 bytes。
