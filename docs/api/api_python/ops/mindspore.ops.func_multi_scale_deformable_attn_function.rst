mindspore.ops.multi_scale_deformable_attn_function
========================================================

.. py:function:: mindspore.ops.multi_scale_deformable_attn_function(value, shape, offset, locations, weight)

    多尺度可变形注意力机制，将多个视角的特征图进行融合。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **value** (Tensor) - 特征张量，数据类型为 ``'float32'`` ， ``'float16'`` 。shape为 :math:`(bs, num\_keys, num\_heads, embed\_dims)`。其中 ``'bs'`` 为 ``'batch size'`` ， ``'num_keys'`` 为特征图的大小， ``'num_heads'`` 为头的数量， ``'embed_dims'`` 为特征图的维度，其中 ``'embed_dims'`` 需要为 ``8`` 的倍数。
        - **shape** (Tensor) - 特征图的形状，数据类型为 ``'int32'`` ， ``'int64'`` 。shape为 :math:`(num\_levels, 2)`。其中 ``'num_levels'`` 为特征图的数量， ``2`` 分别代表 ``'H, W'`` 。
        - **offset** (Tensor) - 偏移量张量，数据类型为 ``'int32'`` ， ``'int64'`` 。shape为 :math:`(num\_levels)`。
        - **locations** (Tensor) - 位置张量，数据类型为 ``'float32'`` ， ``'float16'`` 。shape为 :math:`(bs, num\_queries, num\_heads, num\_levels, num\_points, 2)`。其中 ``'bs'`` 为 ``'batch size'`` ， ``'num_queries'`` 为查询的数量， ``'num_heads'`` 为头的数量， ``'num_levels'`` 为特征图的数量， ``'num_points'`` 为采样点的数量， ``2`` 分别代表 ``'y, x'`` 。
        - **weight** (Tensor) - 权重张量，数据类型为 ``'float32'`` ， ``'float16'`` 。shape为 :math:`(bs, num\_queries, num\_heads, num\_levels, num\_points)`。其中 ``'bs'`` 为 ``'batch size'`` ， ``'num_queries'`` 为查询的数量， ``'num_heads'`` 为头的数量， ``'num_levels'`` 为特征图的数量， ``'num_points'`` 为采样点的数量。

    返回：
        Tensor，融合后的特征张量，数据类型为 ``'float32'`` ， ``'float16'`` 。shape为 :math:`(bs, num\_queries, num\_heads*embed\_dims)`。

    异常：
        - **RuntimeError** - `value` 的数据类型不为 ``'float32'`` ， ``'float16'`` 。
        - **RuntimeError** - `shape` 的数据类型不为 ``'int32'`` ， ``'int64'`` 。
        - **RuntimeError** - `offset` 的数据类型不为 ``'int32'`` ， ``'int64'`` 。
        - **RuntimeError** - `locations` 的数据类型不为 ``'float32'`` ， ``'float16'`` 。
        - **RuntimeError** - `weight` 的数据类型不为 ``'float32'`` ， ``'float16'`` 。
        - **RuntimeError** - `embed_dims` 不为 ``8`` 的倍数。