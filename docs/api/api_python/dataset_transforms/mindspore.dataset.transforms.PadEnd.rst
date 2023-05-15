mindspore.dataset.transforms.PadEnd
===================================

.. py:class:: mindspore.dataset.transforms.PadEnd(pad_shape, pad_value=None)

    对输入Tensor进行填充，要求 `pad_shape` 与输入Tensor的维度保持一致。

    参数：
        - **pad_shape** (list(int)) - 指定填充的shape。维度设置为 ``None`` 时将不会被填充，设置为较小的维数时该维度的元素将被截断。
        - **pad_value** (Union[str, bytes, int, float, bool], 可选) - 用于填充的值。默认值： ``None`` ，表示不指定填充值。
          当指定为默认值，输入Tensor为数值型时默认填充 ``0`` ，输入Tensor为字符型时填充空字符串。

    异常：      
        - **TypeError** - 参数 `pad_shape` 的类型不为list。
        - **TypeError** - 参数 `pad_value` 的类型不为str、float、bool、int或bytes。
        - **TypeError** - 参数 `pad_shape` 的元素类型不为int。
        - **ValueError** - 参数 `pad_shape` 的元素不为正数。


    
