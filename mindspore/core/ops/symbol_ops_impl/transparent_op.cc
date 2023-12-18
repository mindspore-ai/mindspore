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
#include "mindspore/core/symbolic_shape/operation_builder.h"

namespace mindspore {
namespace symshape {
namespace ops {
REG_SYMBOL_OP_BUILDER("Abs").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("Assign").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("BiasAdd").SetShapeDepend({DependOn::kShape, DependOn::kNone});
REG_SYMBOL_OP_BUILDER("Cast").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("DropoutGrad").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("Exp").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("LogicalNot").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("Log").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("LogSoftmaxGrad").SetShapeDepend({DependOn::kNone, DependOn::kShape});
REG_SYMBOL_OP_BUILDER("LogSoftmax").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("Neg").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("OnesLike").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("Reciprocal").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("ReluGrad").SetShapeDepend({DependOn::kShape, DependOn::kNone});
REG_SYMBOL_OP_BUILDER("ReLU").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("SoftmaxGrad").SetShapeDepend({DependOn::kNone, DependOn::kShape});
REG_SYMBOL_OP_BUILDER("Softmax").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("Sqrt").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("Rsqrt").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("RsqrtGrad").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("Tril").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("Sigmoid").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("SigmoidGrad").SetShapeDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("Square").SetShapeDepend({DependOn::kShape});

REG_SYMBOL_OP_BUILDER("Shape").SetValueDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("TensorShape").SetValueDepend({DependOn::kShape});
REG_SYMBOL_OP_BUILDER("ScalarToTensor").SetValueDepend({DependOn::kValue});
REG_SYMBOL_OP_BUILDER("TensorToTuple").SetValueDepend({DependOn::kValue});
REG_SYMBOL_OP_BUILDER("TupleToTensor").SetValueDepend({DependOn::kValue});
}  // namespace ops
}  // namespace symshape
}  // namespace mindspore
