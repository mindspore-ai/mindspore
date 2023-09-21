mindspore.build_searched_strategy
=======================================

.. py:function:: mindspore.build_searched_strategy(strategy_filename)

    构建网络中每个参数的策略，用于分布式推理。

    参数：
        - **strategy_filename** (str) - 策略文件的名称。

    返回：
        Dict，key为参数名，value为该参数的切片策略。

    异常：
        - **ValueError** - 策略文件不正确。
        - **TypeError** - `strategy_filename` 不是str。
