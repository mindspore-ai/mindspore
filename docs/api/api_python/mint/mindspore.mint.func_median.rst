mindspore.mint.median
======================

.. py:function:: mindspore.mint.median(input, dim=None, keepdim=False)

    输出指定维度 ``dim`` 上的中值与其对应的索引。如果 ``dim`` 为None，则计算Tensor中所有元素的中值。

    参数：
        - **input** (Tensor) - 任意维度的Tensor，支持的数据类型为：uint8, int16, int32, int64, float16 or float32.
        - **dim** (int, 可选) - 进行中值计算的轴。默认值： ``None`` 。
        - **keepdim** (bool, 可选) - 是否保留 ``dim`` 指定的维度。默认值： ``False`` 。

    输出：
        - **y** (Tensor) - 输出中值，数据类型与 ``input`` 相同。

          - 如果 ``dim`` 为 ``None`` ， ``y`` 只有一个元素。
          - 如果 ``keepdim`` 为 ``True`` ， ``y`` 的shape除了在 ``dim`` 维度上为1外与 ``input`` 一致。
          - 其他情况下， ``y`` 比 ``input`` 缺少 ``dim`` 指定的维度。
          
        - **indices** (Tensor) - 中值的索引。shape与 ``y`` 一致，数据类型为int64。

    异常：
        - **TypeError** - ``input`` 不是以下数据类型之一：uint8、int16、int32、int64、float16、float32。
        - **TypeError** - ``input`` 不是Tensor。
        - **TypeError** - ``dim`` 不是int。
        - **TypeError** - ``keepdim`` 不是bool值。
        - **ValueError** - ``dim`` 不在 [-x.dim, x.dim-1] 范围内。

