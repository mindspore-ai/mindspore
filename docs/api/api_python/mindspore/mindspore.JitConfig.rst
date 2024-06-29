mindspore.JitConfig
====================

.. py:class:: mindspore.JitConfig(jit_level="", exc_mode="auto", jit_syntax_level="", debug_level="RELEASE", infer_boost="off", **kwargs)

    编译时所使用的JitConfig配置项。

    参数：
        - **jit_level** (str, 可选) - 用于控制编译优化等级，支持["O0", "O1", "O2"]。默认值： ``""`` ，框架自动选择执行方式。不推荐使用，建议使用jit装饰器。

          - ``"O0"``: 除必要影响功能的优化外，其他优化均关闭，使用逐算子执行的执行方式。
          - ``"O1"``: 使能常用优化和自动算子融合优化，使用逐算子执行的执行方式。
          - ``"O2"``: 开启极致性能优化，使用下沉的执行方式。

        - **exc_mode** (str, 可选) - 用于控制模型的执行方式，目前仅支持 ``"auto"``。默认值： ``"auto"`` 。

          - ``"auto"``: 框架自动选择执行方式。
          - ``"sink"``: 支持网络整体一次下发加载到device上，后续由输入驱动执行，无需逐个算子遍历下发，以便取得更好的执行性能，该模式仅在昇腾后端支持。
          - ``"no_sink"``: 网络模型按照单算子逐个异步下发的方式执行。

        - **jit_syntax_level** (str, 可选) - 设置JIT语法支持级别，其值必须为 ``"STRICT"``, ``"LAX"`` 或 ``""`` 。
          默认是空字符串，表示忽略该项JitConfig配置，将使用ms.context的jit_syntax_level，ms.context请参考
          `set_context <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_context.html>`_ 。
          默认值： ``""`` 。
		  
          - ``"STRICT"``: 仅支持基础语法，且执行性能最佳。可用于MindIR导入导出。
          - ``"LAX"``: 最大程度地兼容Python所有语法。执行性能可能会受影响，不是最佳。由于存在可能无法导出的语法，不能用于MindIR导入导出。

        - **debug_level** (int) - 设置调试过程的配置。其值必须为 ``RELEASE`` 或 ``DEBUG`` 。默认值： ``RELEASE`` 。

          - ``RELEASE`` : 正常场景下使用，一些调试信息会被丢弃以获取一个较好的编译性能。
          - ``DEBUG`` : 当错误发生时，用来调试，在编译过程中，更多的调试信息会被记录下来。

        - **infer_boost** (str, 可选): 使能推理加速模式。
          只能设置为 ``"on"`` 或 ``"off"``。 默认设置为 "off"，表示关闭推理加速。
          当使能了推理加速模式，MindSpore会优先使用高性能算子库，并优化运行时，提高推理性能。
          注意：当前推理加速模式只能在 `jit_level` 设为 ``"O0"`` 时使用，且仅支持Atlas A2系列产品。

        - **kwargs** (dict) - 关键字参数字典。
