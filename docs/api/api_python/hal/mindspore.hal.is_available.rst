mindspore.hal.is_available
=============================

.. py:function:: mindspore.hal.is_available(device_target)

    查询指定后端是否可用。
    若指定后端是可用的，那么所有依赖库需要被成功加载。

    参数：
        - **device_target** (str) - 用户指定的后端类型，必须是 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 的其中一个。


    返回：
        bool，指定后端在当前MindSpore包中是否可用。
