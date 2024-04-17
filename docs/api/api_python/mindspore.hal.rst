mindspore.hal
=============

Hal中封装了设备管理、流管理与事件管理的接口。MindSpore从不同后端抽象出对应的上述模块，允许用户在Python层调度硬件资源。

设备管理
------------

.. mscnplatformautosummary::
    :toctree: hal
    :nosignatures:
    :template: classtemplate.rst

    mindspore.hal.device_count
    mindspore.hal.get_arch_list
    mindspore.hal.get_device_capability
    mindspore.hal.get_device_name
    mindspore.hal.get_device_properties
    mindspore.hal.is_available
    mindspore.hal.is_initialized

流管理
------------

.. mscnplatformautosummary::
    :toctree: hal
    :nosignatures:
    :template: classtemplate.rst

    mindspore.hal.current_stream
    mindspore.hal.default_stream
    mindspore.hal.set_cur_stream
    mindspore.hal.synchronize
    mindspore.hal.Stream
    mindspore.hal.StreamCtx

事件管理
------------

.. mscnplatformautosummary::
    :toctree: hal
    :nosignatures:
    :template: classtemplate.rst

    mindspore.hal.Event
