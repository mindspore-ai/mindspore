/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#ifndef CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H
#define CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H

#include "op_proto_macro.h"

namespace ge {
/**
 * @brief Sparse update '*var' as FOBOS algorithm with fixed learning rate.
 * prox_v = var - alpha
 * var = sign(prox_v)/(1+alpha * l2) * max{|prox_v|-alpha * l1,0}
 *
 * @attention Constraints:
 * the input tensors expect indices must have the same shape.
 *
 * @par Inputs:
 * @li var: A mutable tensor. Should be from a Variable().
 * @li alpha: A scalar. Has the same type as "var".
 * @li l1: A scalar. Has the same type as "var".
 * @li l2: A scalar. Has the same type as "var".
 * @li grad: A tensor for the gradient. Has the same type as "var".
 * @li indices: A vector of indices into the first dimension of "var".
 *
 * @par Attributes:
 * use_locking: An optional bool. Defaults to "False".
 *     If "True", updating of the "var", "ms", and "mom" tensors is protected
 *     by a lock; otherwise the behavior is undefined, but may exhibit less
 *     contention.
 *
 * @par Outputs:
 * var: A mutable tensor. Has the same type as input "var".
 *
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator SparseApplyProximalGradientDescent.
 *
 */
REG_CUST_OP(SparseApplyProximalGradientDescent)
  .INPUT(var, TensorType::NumberType())
  .INPUT(alpha, TensorType::NumberType())
  .INPUT(l1, TensorType::NumberType())
  .INPUT(l2, TensorType::NumberType())
  .INPUT(grad, TensorType::NumberType())
  .INPUT(indices, TensorType::IndexNumberType())
  .OUTPUT(var, TensorType::NumberType())
  .ATTR(use_locking, Bool, false)
  .CUST_OP_END_FACTORY_REG(SparseApplyProximalGradientDescent)
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H