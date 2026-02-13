mindspore.device_context.ascend.op_tuning.op_compile
====================================================

.. py:function:: mindspore.device_context.ascend.op_tuning.op_compile(value)

    是否选择在线编译，框架默认设置为静态shape选择在线编译，动态shape选择算子二进制文件，该默认设置将来可能会发生变化。
    如果您想了解更多详细信息，
    请查询 `昇腾社区 <https://www.hiascend.com/document/detail/zh/canncommercial/80RC3/apiref/appdevgapi/aclcppdevg_03_1371.html>`_ 了解。

    参数：
        - **value** (bool) - 表示是否选择在线编译。

          - ``True``：优先选择在线编译。
          - ``False``：优先选择系统中已经编译好的算子二进制文件，提升编译性能。