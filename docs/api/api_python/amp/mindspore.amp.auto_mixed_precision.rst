mindspore.amp.auto_mixed_precision
==================================

.. py:function:: mindspore.amp.auto_mixed_precision(network, amp_level="O0", dtype=mstype.float16)

    返回一个经过自动混合精度处理的网络。

    该接口会对输入网络进行自动混合精度处理，处理后的网络里的Cell和算子增加了精度转换操作，以低精度进行计算，如 ``mstype.float16`` 或 ``mstype.bfloat16`` 。
    Cell和算子的输入和参数被转换成低精度浮点数，计算结果被转换回全精度浮点数，即  ``mstype.float32`` 。

    `amp_level` 及其对应名单决定了哪些Cell和算子需要进行精度转换。

    当 `amp_level` 配置为 ``O0`` 时，不对Cell和算子进行精度转换。

    当 `amp_level` 配置为 ``O1`` 时，白名单内的Cell和算子会被转换为低精度运算。白名单的具体内容可参考 :func:`mindspore.amp.get_white_list` 。

    当 `amp_level` 配置为 ``O2`` 时，黑名单内的Cell保持全精度运算，名单外的Cell会被转换为低精度运算。黑名单的具体内容可参考 :func:`mindspore.amp.get_black_list` 。

    当 `amp_level` 配置为 ``O3`` 时，所有Cell和算子都转换为低精度运算。

    当 `amp_level` 配置为 ``auto`` 时， `auto_whitelist` 名单里的算子会被转换为低精度运算， `auto_blacklist` 名单里的算子会被转换为全精度运算， `promote_list` 名单里的算子会被转换为算子输入中最高精度的浮点类型，名单外的算子使用输入的类型进行计算。

    `auto_whitelist` 名单里的算子包括：

    ``Conv2D`` 、 ``Conv2DExt`` 、 ``Conv3D`` 、 ``Conv3DExt`` 、 ``Conv2DTranspose`` 、 ``ConvTranspose2D`` 、 ``Conv3DTranspose`` 、 ``Convolution`` 、 ``MatMul`` 、 ``MatMulExt`` 、 ``BatchMatMul`` 、 ``BatchMatMulExt`` 、 ``PReLU`` 、 ``Einsum`` 、 ``Dense`` 、 ``Addmm`` 、 ``Addbmm`` 、 ``Addmv`` 、 ``Baddbmm`` 、 ``Mv``

    `auto_blacklist` 名单里的算子包括：

    ``Pow`` 、 ``ACos`` 、 ``Asin`` 、 ``Cosh`` 、 ``Erfinv`` 、 ``Exp`` 、 ``Expm1`` 、 ``Log`` 、 ``Log10`` 、 ``Log1p`` 、 ``Log2`` 、 ``Reciprocal`` 、 ``Rsqrt`` 、 ``Sinh`` 、 ``Tan`` 、 ``Softplus`` 、 ``SoftplusExt`` 、 ``LayerNorm`` 、 ``LayerNormExt`` 、 ``BatchNorm`` 、 ``BatchNormExt`` 、 ``GroupNorm`` 、 ``KLDivLoss`` 、 ``SmoothL1Loss`` 、 ``MultilabelMarginLoss`` 、 ``SoftMarginLoss`` 、 ``TripletMarginLoss`` 、 ``MultiMarginLoss`` 、 ``BCEWithLogitsLoss`` 、 ``Pdist`` 、 ``Cdist`` 、 ``Renorm`` 、 ``ReduceProd`` 、 ``Softmax`` 、 ``LogSoftmax`` 、 ``LogSoftmaxExt`` 、 ``CumProd`` 、 ``CumSum`` 、 ``CumsumExt`` 、 ``ProdExt`` 、 ``SumExt`` 、 ``Norm`` 、 ``L1LossExt`` 、 ``MSELossExt`` 、 ``NLLLoss`` 、 ``NLLLoss2d``

    `promote_list` 名单里的算子包括：

    ``Addcdiv`` 、 ``Addcmul`` 、 ``Cross`` 、 ``_PyboostCrossPrim`` 、 ``Dot`` 、 ``GridSampler2D`` 、 ``GridSampler3D`` 、 ``BiasAdd`` 、 ``AddN`` 、 ``Concat``

    .. note::
        - 重复调用混合精度接口，如 `custom_mixed_precision` 和 `auto_mixed_precision` ，可能导致网络层数增大，性能降低。
        - 如果使用 :class:`mindspore.train.Model` 和 :func:`mindspore.amp.build_train_network` 等接口来训练经\
          过 `custom_mixed_precision` 和 `auto_mixed_precision` 等混合精度接口转换后的网络，则需要将 `amp_level` 配置\
          为 ``O0`` 以避免重复的精度转换。
        - 当 `amp_level` 配置为 ``auto`` 时，网络输出的类型可能是低精度类型，此时可能需要手动转换类型以避免loss函数出现类型不一致的报错。
        - 当 `amp_level` 配置为 ``auto`` ，而网络里的Cell配置了 `to_float` 时， `to_float` 指定的精度优先生效。

    .. warning::
        ``auto`` 等级的 `amp_level` 是实验性API，后续可能修改或删除。

    参数：
        - **network** (Union[Cell, function]) - 定义网络结构。仅当 `amp_level` 配置为 ``auto`` 时支持Function类型。
        - **amp_level** (str, 可选) - 支持["O0", "O1", "O2", "O3", "auto"]。默认值： ``"O0"`` 。

          - **"O0"** - 不变化。
          - **"O1"** - 仅将白名单内的Cell和算子转换为低精度运算，其余部分保持全精度运算。
          - **"O2"** - 黑名单内的Cell和算子保持全精度运算，其余部分都转换为低精度运算。
          - **"O3"** - 将网络全部转为低精度运算。
          - **"auto"** - 将 `auto_whitelist` 名单内的算子转换为低精度运算， `auto_blacklist` 名单内的算子转换为全精度运算，
            `promote_list` 名单内的算子转换为算子输入中最高精度的浮点类型，名单外的算子使用输入的类型进行计算。

        - **dtype** (Type, 可选) - 低精度计算时使用的数据类型，可以是 ``mstype.float16`` 或 ``mstype.bfloat16`` 。默认值： ``mstype.float16`` 。

    异常：
        - **TypeError** - `network` 不是Cell或函数。
        - **ValueError** - `amp_level` 不在支持范围内。
        - **ValueError** - `dtype` 既不是 ``mstype.float16`` 也不是 ``mstype.bfloat16`` 。
