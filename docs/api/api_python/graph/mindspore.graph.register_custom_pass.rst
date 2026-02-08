mindspore.graph.register_custom_pass
=======================================

.. py:function:: mindspore.graph.register_custom_pass(pass_name, plugin_so_path, device="all", stage="")

    注册自定义pass，使用pass对图结构进行修改，仅对默认的 ``"ms_backend"`` 后端生效。

    .. warning::
        实验性接口，未来可能变更或移除。

    参数：
        - **pass_name** (str) - 自定义pass名称。
        - **plugin_so_path** (str) - 自定义pass插件的绝对路径，以 ``.so`` 结尾。
        - **device** (str，可选) - pass生效的硬件设备，支持的值： ``"cpu"`` 、``"gpu"`` 、``"ascend"`` 或 ``"all"`` 。默认值：``"all"`` 。
        - **stage** (str，可选) - pass生效的编译阶段，保留字段以供将来使用。默认值： ``""`` 。

    返回：
        bool。若自定义pass注册成功则返回 ``True`` ，否则返回 ``False`` 。
