mindspore.dataset.config.get_seed
==================================

.. py:function:: mindspore.dataset.config.get_seed()

    获取随机数的种子。如果随机数的种子已设置，则返回设置的值，否则将返回 `std::mt19937::default_seed <http://www.cplusplus.com/reference/random/mt19937/>`_ 这个默认种子值。

    返回：
        int，表示种子的随机数量。
