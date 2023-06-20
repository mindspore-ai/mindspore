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
 * @brief Update entries in '*var' and '*accum' according to the proximal adagrad scheme. \n
 *
 * @attention Constraints:
 *  the input tensors expect indices must have the same shape.
 *
 * @par Inputs:
 * Nine inputs, including:
 * @li var: A mutable Tensor. Must be one of the following types:
 *     TensorType::NumberType(). Should be a Variable Tensor.
 * @li gradient_accumulator: A mutable Tensor. Must have the same
 *     type as "var". Should be a Variable Tensor.
 * @li gradient_squared_accumulator: A mutable Tensor of the same type as "var".
 *     Should be a Variable Tensor.
 * @li grad: A Tensor of the same type as "var", for the gradient.
 * @li indices: A Tensor of the type of int32 or int64.
 *     A vector of indices into the first dimension of var and accum.
 * @li lr: A Tensor of the same type as "var".
 *     Scaling factor. Must be a scalar.
 * @li l1: A Tensor of the same type as "var".
 *     L1 regulariation. Must be a scalar.
 * @li l2: A Tensor of the same type as "var".
 *     L2 regulariation. Must be a scalar.
 * @li global_step: A Tensor of type int32 or int64.
 *     Training step number. Must be a scalar . \n
 *
 * @par Attributes:
 * use_locking: An optional bool. Defaults to "False".
 *     If "True", updating of the var and accum tensors will be
 *     protected by a lock; otherwise the behavior is undefined,
 *     but may exhibit less contention . \n
 *
 * @par Outputs:
 * var: A mutable Tensor. Has the same type as "var" . \n
 *
 * @par Third-party framework compatibility
 * Compatible with the TensorFlow operator SparseApplyAdagradDA.
 */
REG_CUST_OP(SparseApplyAdagradDA)
  .INPUT(var, TensorType::RealNumberType())
  .INPUT(grad_accum, TensorType::RealNumberType())
  .INPUT(grad_square_accum, TensorType::RealNumberType())
  .INPUT(grad, TensorType::RealNumberType())
  .INPUT(indices, TensorType::RealIndexNumberType())
  .INPUT(lr, TensorType::RealNumberType())
  .INPUT(l1, TensorType::RealNumberType())
  .INPUT(l2, TensorType::RealNumberType())
  .INPUT(global_step, TensorType({DT_INT64}))
  .OUTPUT(var, TensorType::RealNumberType())
  .ATTR(use_locking, Bool, false)
  .CUST_OP_END_FACTORY_REG(SparseApplyAdagradDA)
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H