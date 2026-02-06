mindspore.common.np_dtype
=========================

.. py:class:: mindspore.common.np_dtype

    MindSpore的NumPy数据类型。

    `np_dtype` 的实际路径为 `/mindspore/common/np_dtype.py` 。运行以下命令导入环境：

    .. code-block::

        from mindspore.common import np_dtype

    - **数值型**

      ==============================================   =============================
      定义                                              描述
      ==============================================   =============================
      ``bfloat16``                                     NumPy下的 ``bfloat16`` 数据类型。该类型仅用于构造 ``bfloat16`` 类型的Tensor，不保证NumPy下的完整运算能力。仅当运行时的NumPy版本不小于编译时的NumPy版本、且主版本号一致时生效。
      ==============================================   =============================
