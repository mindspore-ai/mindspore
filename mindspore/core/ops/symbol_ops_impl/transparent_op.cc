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
REG_SYMBOL_OP_BUILDER("Abs").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Assign").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("BiasAdd").SetShapeDepend({DependOn::kShape, DependOn::kNone}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Cast").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("ClampScalar")
  .SetShapeDepend({DependOn::kShape, DependOn::kNone, DependOn::kNone})
  .SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Conv2DBackpropInput")
  .SetShapeDepend({DependOn::kNone, DependOn::kNone, DependOn::kValue})
  .SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("DropoutGrad").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Exp").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("GeLU").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("LogicalNot").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Log").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("LogSoftmaxGrad")
  .SetShapeDepend({DependOn::kNone, DependOn::kShape})
  .SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("LogSoftmax").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("mutable").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Neg").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Ones").SetShapeDepend({DependOn::kValue}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("OnesLike").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("PagedAttention").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Reciprocal").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("ReluGrad").SetShapeDepend({DependOn::kShape, DependOn::kNone}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("ReLU").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Rsqrt").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("RsqrtGrad").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("ReshapeAndCache").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Sigmoid").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("SigmoidGrad").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("SiLU").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("SiLUGrad").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Softmax").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("SoftmaxBackward")
  .SetShapeDepend({DependOn::kNone, DependOn::kShape})
  .SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("SoftmaxGrad").SetShapeDepend({DependOn::kNone, DependOn::kShape}).SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Sqrt").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Square").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("StopGradient").SetShapeFunc(TransparentInput);
REG_SYMBOL_OP_BUILDER("Tril").SetShapeFunc(TransparentInput);

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
