mindspore.ops.fused_infer_attention_score
=========================================

.. py:function:: mindspore.ops.fused_infer_attention_score(query, key, value, *, pse_shift=None, atten_mask=None, actual_seq_lengths=None, actual_seq_lengths_kv=None, dequant_scale1=None, quant_scale1=None, dequant_scale2=None, quant_scale2=None, quant_offset2=None, antiquant_scale=None, antiquant_offset=None, key_antiquant_scale=None, key_antiquant_offset=None, value_antiquant_scale=None, value_antiquant_offset=None, block_table=None, query_padding_size=None, kv_padding_size=None, key_shared_prefix=None, value_shared_prefix=None, actual_shared_prefix_len=None, num_heads=1, scale=1.0, pre_tokens=2147483647, next_tokens=2147483647, input_layout='BSH', num_key_value_heads=0, sparse_mode=0, inner_precise=1, block_size=0, antiquant_mode=0, key_antiquant_mode=0, value_antiquant_mode=0, softmax_lse_flag=False)

    这是一个适配增量和全量推理场景的FlashAttention函数，既可以支持全量计算场景（PromptFlashAttention），也可支持增量计算场景（IncreFlashAttention）。
    当Query矩阵的S（Q_S）为1，进入IncreFlashAttention分支，其余场景进入PromptFlashAttention分支。

    .. math::

        Attention(Q,K,V) = Softmax(\frac{QK^{T}}{\sqrt{d}})V

    .. warning::
        - 这是一个实验性API，后续可能修改或删除。
        - 对于Ascend，当前仅支持Atlas A2训练系列产品/Atlas 800I A2推理产品。

    .. note::
        - query、key、value数据排布格式支持从多种维度解读，如下所示：

          - B，Batch size。表示输入样本批量大小。
          - S，Sequence length。表示输入样本序列长度。S1表示query的序列长度，S2表示key/value的序列长度。
          - H，Head size。表示隐藏层的大小。
          - N，Head nums。表示多头数。
          - D，Head dims。表示隐藏层最小的单元尺寸，且满足 :math:`D = H / N`。

    参数：
        - **query** (Tensor) - attention结构的query输入，数据类型支持float16、bfloat16或int8，shape为 :math:`(B, S, H)` ， :math:`(B, N, S, D)`
          或 :math:`(B, S, N, D)` 。
        - **key** (Union[Tensor, tuple[Tensor], list[Tensor]]) - attention结构的key输入，数据类型支持float16、bfloat16或int8，shape为 :math:`(B, S, H)` ，
          :math:`(B, N, S, D)` 或 :math:`(B, S, N, D)` 。
        - **value** (Union[Tensor, tuple[Tensor], list[Tensor]]) - attention结构的value输入，数据类型支持float16、bfloat16或int8，shape为 :math:`(B, S, H)` ，
          :math:`(B, N, S, D)` 或 :math:`(B, S, N, D)` 。

    关键字参数：
        - **pse_shift** (Tensor，可选) - 位置编码参数，数据类型支持float16、bfloat16，默认值为 ``None`` 。

          - 当Q_S不为1时，如果pse_shift为float16类型，则此时的query要求为float16或int8类型。如果pse_shift为bfloat16类型，
            则此时的query要求为bfloat16类型。输入shape类型需为 :math:`(B,N,Q\_S,KV\_S)` 或 :math:`(1,N,Q\_S,KV\_S)` ，其中Q_S为query的
            S轴shape值，KV_S为key和value的S轴shape值。对于pse_shift的KV_S为非32对齐的场景，建议填充到32字节来提高性能，
            多余部分的填充值不做要求。
          - 当Q_S为1时，如果pse_shift为float16类型，则此时的query要求为float16类型。如果pse_shift为bfloat16类型，则此时的query
            要求为bfloat16类型。输入shape类型需为 :math:`(B,N,1,KV\_S)` 或 :math:`(1,N,1,KV\_S)` ，其中KV_S为key和value的S轴shape值。对于
            pse_shift的KV_S为非32对齐的场景，建议填充到32字节来提高性能，多余部分的填充值不做要求。

        - **atten_mask** (Tensor，可选) - 对query乘key的结果进行掩码，数据类型支持int8、uint8和bool。对于每个元素，0表示保留，1表示屏蔽。
          默认值为 ``None`` 。

          - Q_S不为1时建议shape输入为Q_S,KV_S；B,Q_S,KV_S；1,Q_S,KV_S；B,1,Q_S,KV_S；1,1,Q_S,KV_S。
          - Q_S为1时建议shape输入为B,KV_S；B,1,KV_S；B,1,1,KV_S。

        - **actual_seq_lengths** (Union[tuple[int], list[int], Tensor]，可选) - 描述入参query的实际序列长度，数据类型支持int64。如果该参数不指定，可以传入None，
          表示和query的S轴shape值相同。限制：该入参中每个batch的有效序列长度应该不大于query中对应batch的序列长度，Q_S为1时该参数无效。
          默认值为 ``None`` 。
        - **actual_seq_lengths_kv** (Union[tuple[int], list[int], Tensor]，可选) - 描述入参key/value的实际序列长度，数据类型支持int64。如果该参数不指定，可以传入None，
          表示和key/value的S轴shape值相同。限制：该入参中每个batch的有效序列长度应该不大于key/value中对应batch的序列长度，Q_S为1时该参数无效。
          默认值为 ``None`` 。
        - **dequant_scale1** (Tensor，可选) - BMM1后面反量化的量化因子，数据类型支持uint64。支持per-tensor模式。如不使用该功能时可传入None。
          默认值为 ``None`` 。
        - **quant_scale1** (Tensor，可选) - BMM2前面量化的量化因子，数据类型支持float32。支持per-tensor模式。如不使用该功能时可传入None。
          默认值为 ``None`` 。
        - **dequant_scale2** (Tensor，可选) - BMM2后面量化的量化因子，数据类型支持uint64。支持per-tensor模式。如不使用该功能时可传入None。
          默认值为 ``None`` 。
        - **quant_scale2** (Tensor，可选) - 输出量化的量化因子，数据类型支持float32、float16。支持per-tensor、per-channel模式。
          如不使用该功能时可传入None。默认值为 ``None`` 。
        - **quant_offset2** (Tensor，可选) - 输出量化的量化偏移，数据类型支持float32、float16。支持per-tensor、per-channel模式。
          如不使用该功能时可传入None。默认值为 ``None`` 。

          输入为int8，输出为int8的场景：入参dequant_scale1、quant_scale1、dequant_scale2、quant_scale2需要同时存在，
          quant_offset2可选，不传时默认为0。

          - 输出为int8，quant_scale2和quant_offset2为per-channel时，暂不支持左padding、Ring Attention或者D非32Byte对齐的场景。
          - 输出为int8时，暂不支持sparse为band且pre_tokens/next_tokens为负数。
          - 输出为int8，入参quant_offset2传入非空指针和非空tensor值，并且sparse_mode、pre_tokens和next_tokens满足以下条件，
            矩阵会存在某几行不参与计算的情况，导致计算结果误差，该场景会拦截（解决方案：如果希望该场景不被拦截，需要在FIA接口外部做后量化操作，不在FIA接口内部使能）：

            - sparse_mode = 0，atten_mask如果非空指针，每个batch actual_seq_lengths — actual_seq_lengths_kv - pre_tokens > 0 或 next_tokens < 0 时，满足拦截条件；
            - sparse_mode = 1 或 2，不会出现满足拦截条件的情况；
            - sparse_mode = 3，每个batch actual_seq_lengths_kv - actual_seq_lengths < 0，满足拦截条件；
            - sparse_mode = 4，pre_tokens < 0 或每个batch next_tokens + actual_seq_lengths_kv - actual_seq_lengths < 0 时，满足拦截条件。

          输入为int8，输出为float16的场景：入参dequant_scale1、quant_scale1、dequant_scale2需要同时存在。

          输入全为float16或bfloat16，输出为int8的场景：入参quant_scale2需存在，quant_offset2可选，不传时默认为0。

          入参quant_scale2和quant_offset2支持per-tensor/per-channel两种格式和float32/bfloat16两种数据类型。
          若传入quant_offset2，需保证其类型和shape信息与quant_scale2一致。当输入为bfloat16时，同时支持float32和bfloat16，
          否则仅支持float32。per-channel格式，当输出layout为BSH时，要求quant_scale2所有维度的乘积等于H；其他layout要求乘积等于N*D。
          （建议输出layout为BSH时，quant_scale2 shape传入 :math:`(1,1,H)` 或 :math:`(H)` ；输出为BNSD时，建议传入 :math:`(1,N,1,D)`
          或 :math:`(N,D)` ；输出为BSND时，建议传入 :math:`(1,1,N,D)` 或 :math:`(N,D)` ）。

        - **antiquant_scale** (Tensor，可选) - 表示反量化因子，数据类型支持float16、float32或bfloat16。Q_S大于1时只支持float16。
          支持per-tensor、per-channel、per-token模式。默认值为 ``None`` 。
        - **antiquant_offset** (Tensor，可选) - 表示反量化偏移，数据类型支持float16、float32或bfloat16。Q_S大于1时只支持float16。
          支持per-tensor、per-channel、per-token模式。默认值为 ``None`` 。

          antiquant_scale和antiquant_offset参数约束：

          - 支持per-channel、per-tensor和per-token三种模式：

            - per-channel模式：两个参数BNSD场景下shape为 :math:`(2, N, 1, D)` ，BSND场景下shape为 :math:`(2, N, D)` ，
              BSH场景下shape为 :math:`(2, H)` ，其中，2分别对应到key和value，N为num_key_value_heads。参数数据类型和query数据类型相同，antiquant_mode置0。
            - per-tensor模式：两个参数的shape均为 :math:`(2)` ，数据类型和query数据类型相同，antiquant_mode置0。
            - per-token模式：两个参数的shape均为 :math:`(2, B, S)` ，数据类型固定为float32，antiquant_mode置1。

          - 支持对称量化和非对称量化：

            - 非对称量化模式下，antiquant_scale和antiquant_offset参数需同时存在。
            - 对称量化模式下，antiquant_offset可以为空（即 ``None`` ）；当antiquant_offset参数为空时，执行对称量化，否则执行非对称量化。

        - **key_antiquant_scale** (Tensor，可选) - kv伪量化参数分离时表示key的反量化因子，数据类型支持float16、float32或bfloat16。
          支持per-tensor、per-channel、per-token模式。默认值为 ``None`` 。Q_S大于1时该参数无效。
        - **key_antiquant_offset** (Tensor，可选) - kv伪量化参数分离时表示key的反量化偏移，数据类型支持float16、float32或bfloat16。
          支持per-tensor、per-channel、per-token模式。默认值为 ``None`` 。Q_S大于1时该参数无效。
        - **value_antiquant_scale** (Tensor，可选) - kv伪量化参数分离时表示value的反量化因子，数据类型支持float16、float32或bfloat16。
          支持per-tensor、per-channel、per-token模式。默认值为 ``None`` 。Q_S大于1时该参数无效。
        - **value_antiquant_offset** (Tensor，可选) - kv伪量化参数分离时表示value的反量化偏移，数据类型支持float16、float32或bfloat16。
          支持per-tensor、per-channel、per-token模式。默认值为 ``None`` 。Q_S大于1时该参数无效。
        - **block_table** (Tensor，可选) - PageAttention中KV存储使用的block映射表，数据类型支持int32。如不使用该功能时可传入None。
          默认值为 ``None`` 。Q_S大于1时该参数无效。
        - **query_padding_size** (Tensor，可选) - query填充尺寸，数据类型支持int64。表示query中每个batch的数据是否右对齐，且右对齐的个数是多少。
          默认值为 ``None`` 。仅支持Q_S大于1，其余场景该参数无效。
        - **kv_padding_size** (Tensor，可选) - key/value填充尺寸，数据类型支持int64。表示key/value中每个batch的数据是否右对齐，且右对齐的个数是多少。
          默认值为 ``None`` 。仅支持Q_S大于1，其余场景该参数无效。
        - **key_shared_prefix** (Tensor，可选) - key的公共前缀输入。预留参数，暂未启用。默认值为 ``None`` 。
        - **value_shared_prefix** (Tensor，可选) - value的公共前缀输入。预留参数，暂未启用。默认值为 ``None`` 。
        - **actual_shared_prefix_len** (Union[tuple[int], list[int], Tensor]，可选) - 描述公共前缀的实际长度。预留参数，暂未启用。默认值为 ``None`` 。
        - **num_heads** (int，可选) - query的head个数，当input_layout为BNSD时和query的N轴shape值相同。默认值为 ``1`` 。
        - **scale** (double，可选) - 缩放系数，作为计算流中Muls的scalar值。通常，其值为 :math:`1.0 / \sqrt{d}` 。默认值为 ``1.0`` 。
        - **pre_tokens** (int，可选) - 用于稀疏计算，表示attention需要和前多少个token计算关联。默认值为 ``2147483647`` 。Q_S为1时该参数无效。
        - **next_tokens** (int，可选) - 用于稀疏计算，表示attention需要和后多少个token计算关联。默认值为 ``2147483647`` 。Q_S为1时该参数无效。
        - **input_layout** (str，可选) - 指定输入query、key、value的数据排布格式。当前支持BSH、BSND、BNSD、BNSD_BSND。当input_layout为BNSD_BSND时，
          表示输入格式为BNSD，输出格式为BSND，仅支持Q_S大于1。默认值为 ``BSH`` 。
        - **num_key_value_heads** (int，可选) - key/value的head个数，用于支持GQA（Grouped-Query Attention，分组查询注意力）场景。默认值为 ``0`` 。
          0值意味着和key/value的head个数相等。num_heads必须要能够整除num_key_value_heads。num_heads和num_key_value_heads的比值不能大于64。
          当input_layout为BNSD时，该参数需和key/value的N轴shape值相同，否则执行异常。
        - **sparse_mode** (int，可选) - 指定稀疏模式，默认值为 ``0`` 。Q_S为1时该参数无效。

          - 0：代表defaultMask模式。如果atten_mask未传入则不做mask操作，忽略pre_tokens和next_tokens（内部赋值为INT_MAX）；如果传入，则需要传入
            完整的atten_mask矩阵（S1 * S2），表示pre_tokens和next_tokens之间的部分需要计算。
          - 1：代表allMask，必须传入完整的atten_mask矩阵（S1 * S2）。
          - 2：代表leftUpCausal模式的mask，需传入优化后的atten_mask矩阵（2048*2048）。
          - 3：代表rightDownCausal模式的mask，对应以右顶点为划分的下三角场景，需传入优化后的atten_mask矩阵（2048*2048）。
          - 4：代表band模式的mask，即计算pre_tokens和next_tokens之间的部分。需传入优化后的atten_mask矩阵（2048*2048）。
          - 5：代表prefix场景，暂不支持。
          - 6：代表global场景，暂不支持。
          - 7：代表dilated场景，暂不支持。
          - 8：代表block_local场景，暂不支持。

        - **inner_precise** (int，可选) - 共4种模式：0、1、2、3，通过2个bit位表示：第0位（bit0）表示高精度或者高性能选择，第1位（bit1）表示是否做行无效修正。

          - 0：代表开启高精度模式，且不做行无效修正。
          - 1：代表高性能模式，且不做行无效修正。
          - 2：代表开启高精度模式，且做行无效修正。
          - 3：代表高性能模式，且做行无效修正。

          当Q_S>1时，如果sparse_mode为0或1，并传入用户自定义mask，则建议开启行无效；当Q_S为1时，该参数只能为0和1。默认值为 ``1`` 。

          高精度和高性能只对float16生效，行无效修正对float16、bfloat16和int8均生效。
          当前0、1为保留配置值，当计算过程中“参与计算的mask部分”存在某整行全为1的情况时，精度可能会有损失。此时可以尝试将该参数配置为2或3来使能行无效功能以提升精度，但是该配置会导致性能下降。
          如果函数可判断出存在无效行场景，会自动使能无效行计算，例如sparse_mode为3，Sq > Skv场景。

        - **block_size** (int，可选) - PageAttention中KV存储每个block中最大的token个数。默认值为 ``0`` 。Q_S大于1时该参数无效。
        - **antiquant_mode** (int，可选) - 伪量化的方式，传入0时表示为per-channel（per-channel包含per-tensor），传入1时表示为per-token。per-channel
          和per-tensor可以通过入参shape的维数区分，如果shape的维数为1，则运行在per-tensor模式；否则运行在per-channel模式。
          默认值为 ``0`` 。Q_S大于1时该参数无效。
        - **key_antiquant_mode** (int，可选) - key的伪量化的方式，传入0时表示为per-channel（per-channel包含per-tensor），传入1时表示为per-token。
          默认值为 ``0`` 。Q_S大于1时该参数无效。
        - **value_antiquant_mode** (int，可选) - value的伪量化的方式，传入0时表示为per-channel（per-channel包含per-tensor），传入1时表示为per-token。
          默认值为 ``0`` 。Q_S大于1时该参数无效。
        - **softmax_lse_flag** (bool，可选) - 是否输出softmax_lse。默认值为 ``False`` 。

    返回：
        attention_out (Tensor)，注意力分值，数据类型为float16、bfloat16或int8。当input_layout为BNSD_BSND时，输出的shape为 :math:`(B, S, N, D)` 。
        其余情况输出的shape和query的shape一致。

        softmax_lse (Tensor)，softmax_lse值，数据类型为float32，通过将query乘key的结果经过lse（log、sum和exp）计算后获得。具体地，ring attention算法对query乘
        key的结果先取max，得到softmax_max；query乘key的结果减去softmax_max，再取exp，最后取sum，得到softmax_sum；最后对softmax_sum取log，再加上
        softmax_max，得到softmax_lse。softmax_lse只在softmax_lse_flag为True时计算，输出的shape为 :math:`(B, N, Q\_S, 1)` 。如果softmax_lse_flag
        为False，则将返回一个shape为 :math:`(1)` 的全0的Tensor。在图模式且GE后端场景下请确保使能softmax_lse_flag后再使用softmax_lse，否则将遇到异常。

    约束：
        - 全量推理场景（Q_S > 1）：

          - query，key，value输入，功能使用限制如下：

            - 支持B轴小于等于65535，输入类型包含int8时D轴非32对齐或输入类型为float16或bfloat16时D轴非16对齐时，B轴仅支持到128。
            - 支持N轴小于等于256，支持D轴小于等于512。
            - S支持小于等于20971520（20M）。部分长序列场景下，如果计算量过大可能会导致pfa算子执行超时（AICore error类型报错，errorStr为：timeout or trap error），
              此场景下建议做S切分处理，注：这里计算量会受B、S、N、D等的影响，值越大计算量越大。典型的会超时的长序列（即B、S、N、D的乘积较大）场景包括但不限于：

              1. B=1, Q_N=20, Q_S=2097152, D=256, KV_N=1, KV_S=2097152；
              2. B=1, Q_N=2, Q_S=20971520, D=256, KV_N=2, KV_S=20971520；
              3. B=20, Q_N=1, Q_S=2097152, D=256, KV_N=1, KV_S=2097152；
              4. B=1, Q_N=10, Q_S=2097152, D=512, KV_N=1, KV_S=2097152。

            - query、key、value或attention_out类型包含int8时，D轴需要32对齐；类型全为float16、bfloat16时，D轴需16对齐。

          - 参数sparse_mode当前仅支持值为0、1、2、3、4的场景，取其它值时会报错：

            - sparse_mode = 0时，atten_mask如果为None，或者在左padding场景传入atten_mask，则忽略入参pre_tokens、next_tokens。
            - sparse_mode = 2、3、4时，atten_mask的shape需要为S,S或1,S,S或1,1,S,S，其中S的值需要固定为2048，且需要用户保证传入的atten_mask为下三角矩阵，
              不传入atten_mask或者传入的shape不正确报错。
            - sparse_mode = 1、2、3的场景忽略入参pre_tokens、next_tokens并按照相关规则赋值。

          - kvCache反量化仅支持query为float16时，将int8类型的key和value反量化到float16。入参key/value的datarange与入参antiquant_scale的datarange
            乘积范围在（-1，1）范围内，高性能模式可以保证精度，否则需要开启高精度模式来保证精度。

          - query左padding场景：

            - query左padding场景query的搬运起点计算公式为：Q_S - query_padding_size - actual_seq_lengths。query的搬运终点计算公式为：Q_S - query_padding_size。
              其中query的搬运起点不能小于0，终点不能大于Q_S，否则结果将不符合预期。
            - query左padding场景kv_padding_size小于0时将被置为0。
            - query左padding场景需要与actual_seq_lengths参数一起使能，否则默认为query右padding场景。
            - query左padding场景不支持PageAttention，不能与block_table参数一起使能。

          - kv左padding场景：

            - kv左padding场景key和value的搬运起点计算公式为：KV_S - kv_padding_size - actual_seq_lengths_kv。key和value的搬运终点计算公式为：
              KV_S - kv_padding_size。其中key和value的搬运起点不能小于0，终点不能大于KV_S，否则结果将不符合预期。
            - kv左padding场景kv_padding_size小于0时将被置为0。
            - kv左padding场景需要与actual_seq_lengths_kv参数一起使能，否则默认为kv右padding场景。
            - kv左padding场景不支持PageAttention，不能与block_table参数一起使能。

          - pse_shift功能使用限制如下：

            - 支持query数据类型为float16或bfloat16或int8场景下使用该功能。
            - query数据类型为float16且pse_shift存在时，强制走高精度模式，对应的限制继承自高精度模式的限制。
            - Q_S需大于等于query的S长度，KV_S需大于等于key的S长度。

          - kv伪量化参数分离当前暂不支持。

        - 增量推理场景（Q_S = 1）：

          - query，key，value输入，功能使用限制如下：

            - 支持B轴小于等于65536，支持N轴小于等于256，支持D轴小于等于512。
            - query、key、value输入类型均为int8的场景暂不支持。

          - page attention场景：

            - page attention的使能必要条件是block_table存在且有效，同时key、value是按照block_table中的索引在一片连续内存中排布，
              支持key、value dtype为float16/bfloat16/int8，在该场景下key、value的input_layout参数无效。
            - block_size是用户自定义的参数，该参数的取值会影响page attention的性能，在使能page attention场景下，block_size需要传入非0值,
              且block_size最大不超过512。key、value输入类型为float16/bfloat16时需要16对齐，key、value输入类型为int8时需要32对齐，推荐使用128。
              通常情况下，page attention可以提高吞吐量，但会带来性能上的下降。
            - page attention使能场景下，当输入kv cache排布格式为（blocknum, block_size, H），且 num_key_value_heads * D 超过64k时，受硬件指令约束，
              会被拦截报错。可通过使能GQA（减小num_key_value_heads）或调整kv cache排布格式为（blocknum, num_key_value_heads, block_size, D）解决。
            - page attention场景的参数key、value各自对应tensor的shape所有维度相乘不能超过int32的表示范围。

          - page attention的使能场景下，以下场景输入S需要大于等于max_block_num_per_seq * block_size：

            - 使能 Attention mask，如 mask shape为(B, 1, 1, S)。
            - 使能 pse_shift，如 pse_shift shape为(B, N, 1, S)。
            - 使能伪量化 per-token模式：输入参数 antiquant_scale和antiquant_offset的shape均为(2, B, S)。

          - kv左padding场景：

            - kv左padding场景kvCache的搬运起点计算公式为：KV_S - kv_padding_size - actual_seq_lengths。kvCache的搬运终点计算公式为：KV_S - kv_padding_size。
              其中kvCache的搬运起点或终点小于0时，返回数据结果为全0。
            - kv左padding场景kv_padding_size小于0时将被置为0。
            - kv左padding场景需要与actual_seq_lengths参数一起使能，否则默认为kv右padding场景。
            - kv左padding场景需要与atten_mask参数一起使能，且需要保证atten_mask含义正确，即能够正确的对无效数据进行隐藏。否则将引入精度问题。

          - pse_shift功能使用限制如下：

            - pse_shift数据类型需与query数据类型保持一致。
            - 仅支持D轴对齐，即D轴可以被16整除。

          - kv伪量化参数分离：

            - key_antiquant_mode和value_antiquant_mode需要保持一致。
            - key_antiquant_scale和value_antiquant_scale要么都为空，要么都不为空。
            - key_antiquant_offset和value_antiquant_offset要么都为空，要么都不为空。
            - key_antiquant_scale和value_antiquant_scale都不为空时，其shape需要保持一致。
            - key_antiquant_offset和value_antiquant_offset都不为空时，其shape需要保持一致。
