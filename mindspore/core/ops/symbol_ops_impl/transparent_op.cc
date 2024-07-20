/**
 * Copyright 2023 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "mindspore/core/ops/symbol_ops_impl/common.h"

namespace mindspore {
namespace symshape {
namespace ops {
// infer symbolic shape. please add ops in lexicographical order.
REG_SYMBOL_OP_BUILDER("Abs").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("AllReduce").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Assign").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("BiasAdd").SetShapeDepend({DependOn::kShape, DependOn::kNone}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("BroadcastTo")
  .SetShapeDepend({DependOn::kNone, DependOn::kValue})
  .SetShapeFunc(TransValueToShape);
REG_SYMBOL_OP_BUILDER("Cast").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("ClampScalar")
  .SetShapeDepend({DependOn::kShape, DependOn::kNone, DependOn::kNone})
  .SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Conv2DBackpropFilter")
  .SetShapeDepend({DependOn::kNone, DependOn::kNone, DependOn::kValue})
  .SetShapeFunc(TransValueToShape);
REG_SYMBOL_OP_BUILDER("Conv2DBackpropInput")
  .SetShapeDepend({DependOn::kNone, DependOn::kNone, DependOn::kValue})
  .SetShapeFunc(TransValueToShape);
REG_SYMBOL_OP_BUILDER("CudnnUniformReal").SetShapeDepend({DependOn::kValue}).SetShapeFunc(TransValueToShape);
REG_SYMBOL_OP_BUILDER("DropoutGrad").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("DynamicBroadcastTo")
  .SetShapeDepend({DependOn::kNone, DependOn::kValue})
  .SetShapeFunc(TransValueToShape);
REG_SYMBOL_OP_BUILDER("Erf").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Exp").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("FastGeLU").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("FillV2").SetShapeDepend({DependOn::kValue, DependOn::kNone}).SetShapeFunc(TransValueToShape);
REG_SYMBOL_OP_BUILDER("GeLU").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("IsFinite").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("LogicalNot").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Log").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("LogSoftmaxGrad")
  .SetShapeDepend({DependOn::kNone, DependOn::kShape})
  .SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("LogSoftmax").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("mutable").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Neg").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Ones").SetShapeDepend({DependOn::kValue}).SetShapeFunc(TransValueToShape);
REG_SYMBOL_OP_BUILDER("OnesLike").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("PagedAttention").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Reciprocal").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("ReluGrad").SetShapeDepend({DependOn::kShape, DependOn::kNone}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("ReLU").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Rsqrt").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("RsqrtGrad").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("ReshapeAndCache").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Sigmoid").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("SigmoidGrad").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("SiLU").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("SiLUGrad").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Sin").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Softmax").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("SoftmaxBackward")
  .SetShapeDepend({DependOn::kNone, DependOn::kShape})
  .SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("SoftmaxGrad").SetShapeDepend({DependOn::kNone, DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Sqrt").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Square").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("StandardNormal").SetShapeDepend({DependOn::kValue}).SetShapeFunc(TransValueToShape);
REG_SYMBOL_OP_BUILDER("StopGradient").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("StridedSliceGrad")
  .SetShapeDepend({DependOn::kNone, DependOn::kValue})
  .SetShapeFunc(TransValueToShape);
REG_SYMBOL_OP_BUILDER("TensorScatterUpdate").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Tril").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("UniformExt").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Zeros").SetShapeDepend({DependOn::kValue}).SetShapeFunc(TransValueToShape);
REG_SYMBOL_OP_BUILDER("ZerosLike").SetShapeDepend({DependOn::kShape}).SetShapeFunc(TransparentInput);

// infer symbolic value.
REG_SYMBOL_OP_BUILDER("Shape").SetValueDepend({DependOn::kShape}).SetValueFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("TensorShape").SetValueDepend({DependOn::kShape}).SetValueFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("ScalarToTensor").SetValueDepend({DependOn::kValue}).SetValueFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("TensorToTuple").SetValueDepend({DependOn::kValue}).SetValueFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("TupleToTensor").SetValueDepend({DependOn::kValue}).SetValueFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("ListToTuple").SetValueDepend({DependOn::kValue}).SetValueFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("TupleToList").SetValueDepend({DependOn::kValue}).SetValueFunc(TransparentInput);
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
