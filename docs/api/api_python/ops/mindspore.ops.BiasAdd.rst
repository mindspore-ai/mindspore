mindspore.ops.BiasAdd
=====================

.. py:class:: mindspore.ops.BiasAdd(data_format="NCHW")

    返回输入Tensor与偏置Tensor之和。相加前会把偏置Tensor广播成与输入Tensor的shape一致。

    参数：
        - **data_format** (str，可选) - 输入和输出数据的格式。取值为 ``"NHWC"`` 、 ``"NCHW"`` 或 ``"NCDHW"`` 。默认值： ``"NCHW"`` 。

    输入：
        - **input_x** (Tensor) -输入Tensor。shape可以有2~5个维度。支持数据类型：

          - Ascend/CPU： 所有数值类型。
          - GPU： float16、float32、int8。

        - **bias** (Tensor) - 偏置Tensor，shape为 :math:`(C)`。C必须与 `input_x` 的通道维度C相同。其数据类型与 `input_x` 一致。

    输出：
        Tensor，shape和数据类型与 `input_x` 相同。

    异常：
        - **TypeError** - `data_format` 不是str。
        - **ValueError** - `data_format` 的值不在['NHWC', 'NCHW', 'NCDHW']范围内。
        - **TypeError** - `input_x` 或  `bias` 不是Tensor。
        - **TypeError** - `input_x` 与  `bias` 的数据类型不一致。
        - **TypeError** - `input_x` 的维度不在[2, 5]范围内。
