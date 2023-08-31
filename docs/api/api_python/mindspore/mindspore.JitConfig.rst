mindspore.JitConfig
====================

.. py:class:: mindspore.JitConfig(jit_level="O1", exc_mode="auto", jit_syntax_level="", **kwargs)

    编译时所使用的JitConfig配置项。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **jit_level** (str, 可选) - 用于控制编译优化等级，支持["O0", "O1", "O2"]。默认值： ``"O1"`` 。

          - ``"O0"``: 除必要影响功能的优化外，其他优化均关闭。
          - ``"O1"``: 使能常用优化，推荐设置O1级别。
          - ``"O2"``: 开启一些实验等级的优化。

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

        - **kwargs** (dict) - 关键字参数字典。
