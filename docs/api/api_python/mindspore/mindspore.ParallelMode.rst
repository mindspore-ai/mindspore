mindspore.ParallelMode
========================

.. py:class:: mindspore.ParallelMode

    并行模式。

    有五种并行模式，分别是 ``STAND_ALONE`` 、 ``DATA_PARALLEL`` 、 ``HYBRID_PARALLEL`` 、 ``SEMI_AUTO_PARALLEL`` 和 ``AUTO_PARALLEL`` 。默认值： ``STAND_ALONE`` 。

    - ``STAND_ALONE`` ：单卡模式。
    - ``DATA_PARALLEL`` ：数据并行模式。
    - ``HYBRID_PARALLEL`` ：手动实现数据并行和模型并行。
    - ``SEMI_AUTO_PARALLEL`` ：半自动并行模式。
    - ``AUTO_PARALLEL`` ：自动并行模式。

    ``MODE_LIST`` ：所有支持的并行模式列表。