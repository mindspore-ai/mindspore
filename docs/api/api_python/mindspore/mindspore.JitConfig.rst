mindspore.JitConfig
====================

.. py:class:: mindspore.JitConfig(jit_level="O1", task_sink=True, **kwargs)

    编译时所使用的JitConfig配置项。

    .. note::
        - 这是一个实验性接口，后续可能删除或修改。

    参数：
        - **jit_level** (str) - 设置编译优化的级别，支持["O0", "O1", "O2"]。默认值："O1"。

          - "O0": 基础优化。
          - "O1": 手动优化。
          - "O2": 手动优化与图算优化结合。

        - **task_sink** (bool) - 数据是否直接下沉至处理器进行处理。默认值：True。
        - **kwargs** (dict) - 关键字参数字典。
