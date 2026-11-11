Removed legacy nn layers and losses
===================================

The following ten public classes and their re-exports have been removed without compatibility aliases.
The dedicated ``nn.layer.dense`` and ``nn.layer.timedistributed`` modules are also removed.
``nn.Dense``, ``nn.LossBase``, other losses and shared operators remain available.
Update network source, direct imports and class names in configurations together.

.. list-table:: Migration options and limits
   :header-rows: 1
   :widths: 25 75

   * - Removed interface (mindspore.nn prefix)
     - Migration requirements
   * - ``BiDense``
     - Implement the bilinear operation in a custom Cell. For each sample and output channel k,
       compute ``x1 @ W[k] @ x2.T`` and add ``bias[k]`` only if the original layer used a bias.
       Preserve weight shapes, initialization, parameter names and checkpoint mapping.
       A single-input Dense layer is not an equivalent replacement.
   * - ``TimeDistributed``
     - Evaluate calling a shared Cell on each time slice, or reshaping before and after a shared call.
       Check axes, output structure, shared parameters, randomness and stateful layers such as BatchNorm.
       Neither reshape nor vmap is universally equivalent. ``nn.wrap.TimeDistributed`` is also removed.
   * - ``MAELoss``
     - Evaluate ``nn.L1Loss(reduction=...)`` with the same reduction, as used by retained training tests.
       The construct keyword ``label`` becomes ``labels``. Check dtype, broadcasting, reduction precision and gradients.
   * - ``RMSELoss``
     - For floating-point inputs, compose ``ops.sqrt(nn.MSELoss()(logits, labels))`` in a custom Cell.
       Preserve the mean over all elements and broadcasting. Do not silently add epsilon;
       review gradients at zero error and input validation.
   * - ``DiceLoss``, ``MultiClassDiceLoss``
     - Implement the required loss in a custom Cell, preserving squared denominator terms, smooth, activation,
       class weights, ignored classes and class reduction. Common Dice formulas need not be equivalent.
   * - ``FocalLoss``
     - Implement the original softmax classification semantics, label layout, gamma, weights and reduction.
       A sigmoid binary focal loss is not a drop-in replacement.
   * - ``Pad``
     - For zero CONSTANT padding, use ``ops.pad(x, padding, mode="constant", value=0)``.
       Reverse the original sequence of dimension pairs before flattening it; keep the order within each pair.
       For REFLECT, use reflect mode only for supported input ranks and devices. This API does not support
       SYMMETRIC; reflect is not equivalent because it treats edge elements differently. Check edge values and gradients.
   * - ``LRN``
     - If local response normalization is required, implement the original formula in a custom layer.
       Preserve the channel window radius, boundary handling and the meaning of bias, alpha and beta.
       Compare forward values and gradients against the original model.
   * - ``FractionalMaxPool3d``
     - Call ``ops.fractional_max_pool3d`` with the original kernel_size, output_size/output_ratio,
       return_indices and _random_samples. Preserve randomness and index semantics. FractionalMaxPool2d remains.

These lower-level paths describe availability after this change, not a reversal of future API retirement decisions.
Validate forward values, gradients, training updates, checkpoint loading and target-device behavior after migration.
