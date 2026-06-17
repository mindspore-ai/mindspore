mindspore.graph.register_custom_pass
=======================================

.. py:function:: mindspore.graph.register_custom_pass(pass_name, plugin_so_path, device="all", pass_type=CustomPassType.PATTERN)

    注册自定义pass，使用pass对图结构进行修改，仅对默认的 ``"ms_backend"`` 后端生效。

    .. note::
        暂不支持Windows和MacOS系统。用户需要保证自定义pass插件文件的有效性和合法性。

    .. warning::
        实验性接口，未来可能变更或移除。

    参数：
        - **pass_name** (str) - 自定义pass名称。
        - **plugin_so_path** (str) - 自定义pass插件的绝对路径，以 ``.so`` 结尾。
        - **device** (str，可选) - pass生效的硬件设备，支持的值： ``"cpu"`` 、``"gpu"`` 、``"ascend"`` 或 ``"all"`` 。默认值：``"all"`` 。
        - **pass_type** (:class:`~.graph.CustomPassType`, 可选) - 自定义pass类型。默认值： ``CustomPassType.PATTERN`` 。
          可以是 ``CustomPassType.PATTERN`` 、 ``CustomPassType.FULL_GRAPH`` 。
        
          - **CustomPassType.PATTERN**：根据用户想要匹配的图结构修改图结构，如 `Add` 算子和 `Neg` 算子替换为一个 `Sub` 算子，应用于后端优化流程。
          - **CustomPassType.FULL_GRAPH**：基于整图信息修改图结构，应用于前端优化流程。

    返回：
        bool。若自定义pass注册成功则返回 ``True`` ，否则返回 ``False`` 。

    异常：
        - **TypeError** - 如果 `pass_type` 不是 :class:`mindspore.graph.CustomPassType` 类型。
