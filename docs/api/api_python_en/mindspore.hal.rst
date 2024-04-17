mindspore.hal
=============

Hal encapsulates interfaces for device, stream, and event. MindSpore abstracts the corresponding modules from different backends, allowing users to schedule hardware resources at the Python layer.

Device
-----------

.. msplatformautosummary::
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

Stream
---------

.. msplatformautosummary::
    :toctree: hal
    :nosignatures:
    :template: classtemplate.rst

    mindspore.hal.current_stream
    mindspore.hal.default_stream
    mindspore.hal.set_cur_stream
    mindspore.hal.synchronize
    mindspore.hal.Stream
    mindspore.hal.StreamCtx

Event
---------

.. msplatformautosummary::
    :toctree: hal
    :nosignatures:
    :template: classtemplate.rst

    mindspore.hal.Event
