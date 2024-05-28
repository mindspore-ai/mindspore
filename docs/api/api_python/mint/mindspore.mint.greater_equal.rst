mindspore.mint.greater_equal
==============================

.. py:function:: mindspore.mint.greater_equal(input, other)

    给定两个张量（Tensors），逐元素比较它们，以检查第一个张量中的每个元素是否大于或等于第二个张量中的相应元素。

    更多参考详见 :func:`mindspore.ops.ge`。

    参数：
        - **input** (Union[Tensor, Number]) - 第一个输入，是一个Number或数据类型为number或bool_的Tensor。
        - **other** (Union[Tensor, Number]) - 第二个输入，是一个Number或数据类型为number或bool_的Tensor。

    返回：
        Tensor，shape与广播后的shape相同，数据类型为bool。
