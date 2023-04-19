mindspore.pytype_to_dtype
=========================

.. py:function:: mindspore.pytype_to_dtype(obj)

    获取与python数据类型对应的MindSpore数据类型。

    参数：
        - **obj** (type) - Python数据对象。

    返回：
        MindSpore的数据类型。

    异常：
        - **NotImplementedError** - Python类型无法转换为MindSpore类型。