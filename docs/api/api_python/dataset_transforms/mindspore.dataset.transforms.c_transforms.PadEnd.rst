mindspore.dataset.transforms.c_transforms.PadEnd
================================================

.. py:class:: mindspore.dataset.transforms.c_transforms.PadEnd(pad_shape, pad_value=None)

    对输入Tensor进行填充，要求填充值与Tensor的shape一致。

    **参数：**

    - **pad_shape** (list(int)) - 指定填充的shape。维度设置为'None'时将不会被填充，设置为较小的维数时该维度的元素将被截断。
    - **pad_value** (Union[str, bytes, int, float, bool], 可选) - 用于填充的值。默认值：None，未指定填充值。

    **异常：**
      
    - **TypeError** - 参数 `pad_shape` 的类型不为 list。
    - **TypeError** - 参数 `pad_value` 的类型不为 string, float, bool, int 或 bytes。
    - **TypeError** - 参数 `pad_value` 的元素类型不为 int。
    - **ValueError** - 参数 `pad_value` 的元素类型不为正数。


    
