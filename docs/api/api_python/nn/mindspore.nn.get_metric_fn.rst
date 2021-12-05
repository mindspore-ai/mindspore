mindspore.nn.get_metric_fn
===========================

.. py:function:: mindspore.nn.get_metric_fn(name, *args, **kwargs)

    根据输入的 `name` 获取metric的方法。

    **参数：**

    - **name** (str) - metric的方法名，可以通过 :class:`mindspore.nn.names` 接口获取。
    - **args** - metric函数的参数。
    - **kwargs** - metric函数的关键字参数。

    **返回：**

    metric对象，metric方法的类实例。

    **样例：**

    >>> metric = nn.get_metric_fn('precision', eval_type='classification')
