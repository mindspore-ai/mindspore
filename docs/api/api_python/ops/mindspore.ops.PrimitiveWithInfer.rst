mindspore.ops.PrimitiveWithInfer
================================

.. py:class:: mindspore.ops.PrimitiveWithInfer(name)

    PrimitiveWithInfer是Python中的原语基类，在python中定义了跟踪推理的函数。

    可以重写四个方法来定义Primitive的推断逻辑：__infer__()、infer_shape()、infer_dtype()和infer_value()。如果在Primitive中定义了__infer__()，则__infer__()的优先级最高。

    如果未定义__infer__()，则可以定义infer_shape()和infer_dtype()来描述shape和类型的推断逻辑。infer_value()用于常量传播。

    参数：
        - **name** (str) - 当前Primitive的名称。
