mindspore.utils
====================


.. py:function:: mindspore.utils.stress_detect()

    Inspect the hardware to determine if there are any faults affecting its accuracy and precision.Common use cases include invoking this interface at each step or when saving checkpoints, allowing users to check if any hardware issues could impact precision.

    Returns
        int, the return value represents the error type: zero indicates normal operation; non-zero values indicate a hardware failure.
    
    Supported Platforms:
        ``Ascend``
    
    **Examples**
    
        >>> from mindspore.utils import stress_detect
        >>> ret = stress_detect()
        >>> print(ret)
        0
