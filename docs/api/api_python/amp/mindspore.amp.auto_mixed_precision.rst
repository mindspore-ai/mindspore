mindspore.amp.auto_mixed_precision
==================================

.. py:function:: mindspore.amp.auto_mixed_precision(network, amp_level="O0")

    对Cell进行自动混合精度处理。

    参数：
        - **network** (Cell) - 定义网络结构。
        - **amp_level** (str) - 支持["O0", "O1", "O2", "O3"]。默认值："O0"。

          - **"O0"** - 不变化。
          - **"O1"** - 将白名单内的Cell和运算转为float16精度，其余部分保持float32精度。
          - **"O2"** - 将黑名单内的Cell和运算保持float32精度，其余部分转为float16精度。
          - **"O3"** - 将网络全部转为float16精度。

    异常：
        - **ValueError** - `amp_level` 不在支持范围内。
