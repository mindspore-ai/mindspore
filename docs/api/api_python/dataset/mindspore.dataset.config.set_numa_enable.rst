mindspore.dataset.config.set_numa_enable
==========================================

.. py:function:: mindspore.dataset.config.set_numa_enable(numa_enable)

    设置NUMA的默认状态为启动状态。如果 `numa_enable` 为 ``True`` ，则需要确保安装了 `NUMA库 <http://rpmfind.net/linux/rpm2html/search.php?query=libnuma-devel>`_ 。

    参数：
        - **numa_enable** (bool) - 表示是否使用NUMA绑定功能。

    异常：
        - **TypeError** - `numa_enable` 不是bool类型。
