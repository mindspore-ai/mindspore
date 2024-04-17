mindspore.hal.get_arch_list
=============================

.. py:function:: mindspore.hal.get_arch_list(device_target=None)

    返回此MindSpore包支持哪些后端架构。

    .. note::
        - 若用户不指定 `device_target` ，将此参数设置为当前已经设置的后端类型。

    参数：
        - **device_target** (str，可选) - 默认值：None，必须是 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 的其中一个。

    返回：
        GPU后端返回str，Ascend以及CPU后端返回None。
