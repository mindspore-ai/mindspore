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
#include "plugin/device/ascend/kernel/opapi/aclnn_kernel_mod.h"

namespace mindspore {
namespace kernel {
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(Cos, aclnnCos, 2)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(Div, aclnnDiv, 3)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(Equal, aclnnEqTensor, 3)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(Exp, aclnnExp, 2)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(Greater, aclnnGtTensor, 3)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(GreaterEqual, aclnnGeTensor, 3)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(IsFinite, aclnnIsFinite, 2)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(LessEqual, aclnnLeTensor, 3)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(LogicalAnd, aclnnLogicalAnd, 3)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(LogicalNot, aclnnLogicalNot, 2)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(LogicalOr, aclnnLogicalOr, 3)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(LogicalXor, aclnnLogicalXor, 3)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(Log, aclnnLog, 2)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(Maximum, aclnnMaximum, 3)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(Minimum, aclnnMinimum, 3)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(Mul, aclnnMul, 3)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(Neg, aclnnNeg, 2)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(NotEqual, aclnnNeTensor, 3)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(RealDiv, aclnnDiv, 3)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(Reciprocal, aclnnReciprocal, 2)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(Rsqrt, aclnnRsqrt, 2)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(Sigmoid, aclnnSigmoid, 2)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(SigmoidGrad, aclnnSigmoidBackward, 3)
MS_ACLLNN_COMMON_KERNEL_FACTORY_REG(Sqrt, aclnnSqrt, 2)
}  // namespace kernel
}  // namespace mindspore
