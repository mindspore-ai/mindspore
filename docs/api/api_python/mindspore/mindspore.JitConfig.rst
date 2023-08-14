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

        - **jit_syntax_level** (str, 可选) - 设置JIT语法支持级别，支持["STRICT", "LAX"]。
          默认是空字符串，表示忽略该选项。默认值： ``""`` 。
		  
          - "STRICT": 仅支持基础语法，且执行性能最佳。
          - "LAX": 最大程度地兼容Python所有语法。执行性能可能会受影响，不是最佳。

        - **kwargs** (dict) - 关键字参数字典。
