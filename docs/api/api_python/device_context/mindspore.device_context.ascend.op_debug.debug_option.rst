mindspore.device_context.ascend.op_debug.debug_option
=====================================================

.. py:function:: mindspore.device_context.ascend.op_debug.debug_option(option_value)

    使能Ascend算子调试选项，默认不开启。

    参数：
        - **option_value** (str) - Ascend算子调试配置，当前只支持内存访问越界检测，可配置为 ``oom`` 。

          - ``oom`` ：涉及从全局内存中读写数据，例如读写算子数据等，该选项开启全局内存访问越界检测，实际执行算子时，若出现内存越界，AscendCL会返回 ``EZ9999`` 错误码。