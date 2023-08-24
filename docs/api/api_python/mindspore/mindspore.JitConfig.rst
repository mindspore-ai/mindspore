mindspore.JitConfig
====================

.. py:class:: mindspore.JitConfig(jit_level="O1", exc_mode="auto", jit_syntax_level="", **kwargs)

    编译时所使用的JitConfig配置项。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **jit_level** (str, 可选) - 设置编译优化的级别，支持["O0", "O1", "O2", "O3"]。默认值： ``"O1"`` 。

          - "O0": 基础优化。
          - "O1": 手动优化。
          - "O2": 手动优化与图算优化结合。
          - "O3": 性能优化，无法保证泛化性。

        - **exc_mode** (str, 可选) - 设置执行模式，支持["auto", "sink", "no_sink"]。默认值： ``"auto"`` 。

          - "auto": 自动策略。
          - "sink": 计算图下沉策略。
          - "no_sink": 非计算图下沉策略。

        - **jit_syntax_level** (str, 可选) - 设置JIT语法支持级别，其值必须为 ``"STRICT"``, ``"LAX"`` 或 ``""`` 。
          默认是空字符串，表示忽略该项JitConfig配置，将使用ms.context的jit_syntax_level，ms.context请参考
          `set_context <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.set_context.html>`_ 。
          默认值： ``""`` 。
		  
          - "STRICT": 仅支持基础语法，且执行性能最佳。可用于MindIR导入导出。
          - "LAX": 最大程度地兼容Python所有语法。执行性能可能会受影响，不是最佳。由于存在可能无法导出的语法，不能用于MindIR导入导出。

        - **kwargs** (dict) - 关键字参数字典。
