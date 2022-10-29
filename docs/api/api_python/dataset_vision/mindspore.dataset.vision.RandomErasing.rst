mindspore.dataset.vision.RandomErasing
======================================

.. py:class:: mindspore.dataset.vision.RandomErasing(prob=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False, max_attempts=10)

    按照指定的概率擦除输入numpy.ndarray图像上随机矩形区域内的像素。

    请参阅论文 `Random Erasing Data Augmentation <https://arxiv.org/pdf/1708.04896.pdf>`_。

    参数：
        - **prob** (float，可选) - 执行随机擦除的概率，取值范围：[0.0, 1.0]。默认值：0.5。
        - **scale** (Sequence[float, float]，可选) - 擦除区域面积相对原图比例的随机选取范围，按照(min, max)顺序排列，默认值：(0.02, 0.33)。
        - **ratio** (Sequence[float, float]，可选) - 擦除区域宽高比的随机选取范围，按照(min, max)顺序排列，默认值：(0.3, 3.3)。
        - **value** (Union[int, str, Sequence[int, int, int]]) - 擦除区域的像素填充值。若输入int，将以该值填充RGB通道；若输入Sequence[int, int, int]，将分别用于填充R、G、B通道；若输入字符串'random'，将以从标准正态分布获得的随机值擦除各个像素。默认值：0。
        - **inplace** (bool，可选) - 是否直接在原图上执行擦除，默认值：False。
        - **max_attempts** (int，可选) - 生成随机擦除区域的最大尝试次数，超过该次数时将返回原始图像。默认值：10。
    
    异常：        
        - **TypeError** - 当 `prob` 的类型不为float。
        - **TypeError** - 当 `scale` 的类型不为Sequence[float, float]。
        - **TypeError** - 当 `ratio` 的类型不为Sequence[float, float]。
        - **TypeError** - 当 `value` 的类型不为int、str或Sequence[int, int, int]。
        - **TypeError** - 当 `inplace` 的类型不为bool。
        - **TypeError** - 当 `max_attempts` 的类型不为int。
        - **ValueError** - 当 `prob` 取值不在[0.0, 1.0]范围内。
        - **ValueError** - 当 `scale` 为负数。
        - **ValueError** - 当 `ratio` 为负数。
        - **ValueError** - 当 `value` 取值不在[0, 255]范围内。
        - **ValueError** - 当 `max_attempts` 不为正数。
