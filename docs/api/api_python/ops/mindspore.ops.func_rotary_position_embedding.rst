mindspore.ops.rotary_position_embedding
=======================================

.. py:function:: mindspore.ops.rotary_position_embedding(x, cos, sin, mode=0)

    旋转位置编码算法的实现。
    请参阅论文 `Enhanced Transformer with Rotary Position Embedding <https://arxiv.org/pdf/2104.09864.pdf>`_ 。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **x** (Tensor) - 4维Tensor，数据类型float16、bfloat16、float32。
        - **cos** (Tensor) - 4维常量，数据类型和 `x` 一致。取值范围为[-1,1]。
        - **sin** (Tensor) - 和 `cos` 相同。
        - **mode** (int) - 可选属性，选择计算模式。0表示rotate_half（GPT-NeoX style）， 1表示rotate_interleaved（GPT-J style）。默认 ``0`` 。

    .. list-table:: Layout配置约束
        :widths: 5 20 20
        :header-rows: 1

        * - 参数
          - RotateHalf(mode: 0)
          - RotateInterleaved(mode: 1)
        * - x
          - layout支持BNSD、BSND、SBND；

            D < 896，且为2的倍数；B，N < 1000；

          - layout支持：BNSD、BSND、SBND；

            B * N < 1000；D < 896，且D为2的倍数
        * - cos
          - 对应 `x` layout的支持情况：

            `x` 为BNSD：11SD、B1SD、BNSD；

            `x` 为BSND：1S1D、BS1D、BSND；

            `x` 为SBND：S11D、SB1D、SBND。
          - 对应 `x` layout的支持情况：

            `x` 为BNSD：11SD；

            `x` 为BSND：1S1D；

            `x` 为SBND：S11D。
        * - sin
          - 同 `cos`
          - 同 `cos`

    .. note::
        layout是BNSD，并且D是32bytes对齐、B * N > 8S 时，因为性能不好，不允许调用。

    返回：
        Tensor，其shape和dtype和 `x` 保持一致。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `cos` 不是Tensor。
        - **TypeError** - `sin` 不是Tensor。
        - **TypeError** - `mode` 不是int。
