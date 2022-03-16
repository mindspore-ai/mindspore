mindspore.ms_class
==================

.. py:function:: mindspore.class(cls)

    用户自定义类的类装饰器。

    MindSpore可以通过ms_class识别用户定义的类，从而获取这些类的属性和方法。

    **参数：**

    - **cls**  (Class) - 用户自定义的类。

    **返回：**

    带有 __ms_class__ 属性的类。

    **异常：**

    - **TypeError** – 如果 ms_class 用于非 class 类型或者 nn.Cell。
    - **AttributeError** – 如果调用了 ms_class 装饰的类的私有属性或魔术方法。
