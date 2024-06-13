mindspore.Generator
======================

.. py:class:: mindspore.Generator()

    管理随机数状态的生成器，为随机函数提供seed和offset。当seed和offset固定时，随机函数产生相同的随机序列。

    输入：
        - **step** (int) - 偏移量更新的步长。

    输出：
        包含生成器的种子和偏移量的tuple。

    .. py:method:: get_state()

        获取生成器状态。

        返回：
            Tensor，生成器的状态。

    .. py:method:: initial_seed()

        返回生成器的初始种子。

        返回：
            生成器的初始化种子。

    .. py:method:: manual_seed(seed)

        设置生成器种子。

        参数：
            - **seed** (int) - 生成器的种子。

        返回：
            生成器自身。

    .. py:method:: seed()

        生成可作为生成器种子的随机种子。

        返回：
            int类型，随机生成的种子。

    .. py:method:: set_state(state)

        设置生成器状态。

        参数：
            - **state** (tensor) - 生成器的状态。
