mindspore.device_context.ascend.op_debug.aclinit_config
=======================================================

.. py:function:: mindspore.device_context.ascend.op_debug.aclinit_config(config)

    配置aclInit接口的配置项。
    详细信息请查看 `昇腾社区关于aclInit的文档说明 <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/800alpha003/apiref/appdevgapi/aclcppdevg_03_0022.html>`_。

    参数：
        - **config** (dict) - 初始化AscendCL时，可通过该配置接口开启或设置以下功能。

          - ``max_opqueue_num`` ：通过单算子模型方式执行时，为节约内存和平衡调用性能，可通过max_opqueue_num参数配置单算子模型映射队列的最大长度。如果长度达到最大，会先删除长期未使用的映射信息和缓存中的单算子模型，再加载最新的映射信息和对应的单算子模型。如果不配置映射队列的最大长度，则默认最大长度为20000。
          - ``err_msg_mode`` ：用于控制按进程或线程级别获取错误信息，默认按进程级别。"0"表示按线程级别获取错误信息；"1"为默认值，表示按进程级别获取错误信息。
          - ``dump`` ：控制是否开启Ascend算子异常Dump。可选值如下：

            - {"dump_scene": "lite_exception"}，为默认值，表示开启异常dump。
            - {"dump_scene": "lite_exception:disable"}，表示关闭异常Dump。