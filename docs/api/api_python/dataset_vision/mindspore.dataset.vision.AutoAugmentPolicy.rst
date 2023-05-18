mindspore.dataset.vision.AutoAugmentPolicy
==========================================

.. py:class:: mindspore.dataset.vision.AutoAugmentPolicy

    不同数据集的自动增强策略。
    可能的枚举值包括： ``AutoAugmentPolicy.IMAGENET`` 、 ``AutoAugmentPolicy.CIFAR10`` 、 ``AutoAugmentPolicy.SVHN`` 。
    每个策略包含25对增强操作。使用AutoAugment时，每个图像都会使用这些操作对中的一个随机转换。每对有2个不同的操作。下面显示了所有这些增强操作，包括操作名称及其概率和随机参数。

    - ``AutoAugmentPolicy.IMAGENET``：ImageNet的数据集自动增强策略。

      .. code-block::

          Augmentation operations pair:
          [(("Posterize", 0.4, 8), ("Rotate", 0.6, 9)),        (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
           (("Equalize", 0.8, None), ("Equalize", 0.6, None)), (("Posterize", 0.6, 7), ("Posterize", 0.6, 6)),
           (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),    (("Equalize", 0.4, None), ("Rotate", 0.8, 8)),
           (("Solarize", 0.6, 3), ("Equalize", 0.6, None)),    (("Posterize", 0.8, 5), ("Equalize", 1.0, None)),
           (("Rotate", 0.2, 3), ("Solarize", 0.6, 8)),         (("Equalize", 0.6, None), ("Posterize", 0.4, 6)),
           (("Rotate", 0.8, 8), ("Color", 0.4, 0)),            (("Rotate", 0.4, 9), ("Equalize", 0.6, None)),
           (("Equalize", 0.0, None), ("Equalize", 0.8, None)), (("Invert", 0.6, None), ("Equalize", 1.0, None)),
           (("Color", 0.6, 4), ("Contrast", 1.0, 8)),          (("Rotate", 0.8, 8), ("Color", 1.0, 2)),
           (("Color", 0.8, 8), ("Solarize", 0.8, 7)),          (("Sharpness", 0.4, 7), ("Invert", 0.6, None)),
           (("ShearX", 0.6, 5), ("Equalize", 1.0, None)),      (("Color", 0.4, 0), ("Equalize", 0.6, None)),
           (("Equalize", 0.4, None), ("Solarize", 0.2, 4)),    (("Solarize", 0.6, 5), ("AutoContrast", 0.6, None)),
           (("Invert", 0.6, None), ("Equalize", 1.0, None)),   (("Color", 0.6, 4), ("Contrast", 1.0, 8)),
           (("Equalize", 0.8, None), ("Equalize", 0.6, None))]

    - ``AutoAugmentPolicy.CIFAR10``：Cifar10的数据集自动增强策略。

      .. code-block::

          Augmentation operations pair:
          [(("Invert", 0.1, None), ("Contrast", 0.2, 6)),         (("Rotate", 0.7, 2), ("TranslateX", 0.3, 9)),
           (("Sharpness", 0.8, 1), ("Sharpness", 0.9, 3)),         (("ShearY", 0.5, 8), ("TranslateY", 0.7, 9)),
           (("AutoContrast", 0.5, None), ("Equalize", 0.9, None)), (("ShearY", 0.2, 7), ("Posterize", 0.3, 7)),
           (("Color", 0.4, 3), ("Brightness", 0.6, 7)),            (("Sharpness", 0.3, 9), ("Brightness", 0.7, 9)),
           (("Equalize", 0.6, None), ("Equalize", 0.5, None)),     (("Contrast", 0.6, 7), ("Sharpness", 0.6, 5)),
           (("Color", 0.7, 7), ("TranslateX", 0.5, 8)),            (("Equalize", 0.8, None), ("Invert", 0.1, None)),
           (("TranslateY", 0.4, 3), ("Sharpness", 0.2, 6)),        (("Brightness", 0.9, 6), ("Color", 0.2, 8)),
           (("Solarize", 0.5, 2), ("Invert", 0.0, None)),          (("TranslateY", 0.9, 9), ("TranslateY", 0.7, 9)),
           (("Equalize", 0.2, None), ("Equalize", 0.6, None)),     (("Color", 0.9, 9), ("Equalize", 0.6, None)),
           (("AutoContrast", 0.8, None), ("Solarize", 0.2, 8)),    (("Brightness", 0.1, 3), ("Color", 0.7, 0)),
           (("Solarize", 0.4, 5), ("AutoContrast", 0.9, None)),
           (("AutoContrast", 0.9, None), ("Solarize", 0.8, 3)),
           (("TranslateY", 0.7, 9), ("AutoContrast", 0.9, None)),
           (("Equalize", 0.3, None), ("AutoContrast", 0.4, None)),
           (("Equalize", 0.2, None), ("AutoContrast", 0.6, None))]

    - ``AutoAugmentPolicy.SVHN``：SVHN的数据集自动增强策略。

      .. code-block::

          Augmentation operations pair:
          [(("ShearX", 0.9, 4), ("Invert", 0.2, None)),          (("ShearY", 0.9, 8), ("Invert", 0.7, None)),
           (("Equalize", 0.6, None), ("Solarize", 0.6, 6)),      (("Invert", 0.9, None), ("Equalize", 0.6, None)),
           (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),        (("ShearX", 0.9, 4), ("AutoContrast", 0.8, None)),
           (("ShearY", 0.9, 8), ("Invert", 0.4, None)),          (("ShearY", 0.9, 5), ("Solarize", 0.2, 6)),
           (("Invert", 0.9, None), ("AutoContrast", 0.8, None)), (("Equalize", 0.6, None), ("Rotate", 0.9, 3)),
           (("ShearX", 0.9, 4), ("Solarize", 0.3, 3)),           (("ShearY", 0.8, 8), ("Invert", 0.7, None)),
           (("Equalize", 0.9, None), ("TranslateY", 0.6, 6)),    (("Invert", 0.9, None), ("Equalize", 0.6, None)),
           (("Contrast", 0.3, 3), ("Rotate", 0.8, 4)),           (("Invert", 0.8, None), ("TranslateY", 0.0, 2)),
           (("ShearY", 0.7, 6), ("Solarize", 0.4, 8)),           (("Invert", 0.6, None), ("Rotate", 0.8, 4)),
           (("ShearY", 0.3, 7), ("TranslateX", 0.9, 3)),         (("ShearX", 0.1, 6), ("Invert", 0.6, None)),
           (("Solarize", 0.7, 2), ("TranslateY", 0.6, 7)),       (("ShearY", 0.8, 4), ("Invert", 0.8, None)),
           (("ShearX", 0.7, 9), ("TranslateY", 0.8, 3)),         (("ShearY", 0.8, 5), ("AutoContrast", 0.7, None)),
           (("ShearX", 0.7, 2), ("Invert", 0.1, None))]
