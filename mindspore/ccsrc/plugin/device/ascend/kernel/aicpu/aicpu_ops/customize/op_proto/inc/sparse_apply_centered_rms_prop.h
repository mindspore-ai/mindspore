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
* @brief Updates "var" according to the centered RMSProp algorithm.
*  The centered RMSProp algorithm uses an estimate of the centered second moment
*  (i.e., the variance) for normalization, as opposed to regular RMSProp, which
*  uses the (uncentered) second moment. This often helps with training, but is
*  slightly more expensive in terms of computation and memory.
*
*  t-1 mean previous period.
*  mg <- rho * mg{t-1} + (1-rho) * grad
*  ms <- rho * ms{t-1} + (1-rho) * grad * grad
*  mom <- momentum * mom{t-1} + lr * grad / sqrt(ms - mg * mg + epsilon)
*  var <- var - mom
*
* @attention Constraints:
* @li in dense implementation of this algorithm, mg, ms, and mom will
*    update even if the grad is zero, but in this sparse implementation, mg, ms,
*    and mom will not update in iterations during which the grad is zero.
*
* @par Inputs:
* @li var: A mutable tensor. Should be from a Variable().
* @li mg: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
* @li ms: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
* @li mom: A mutable tensor. Has the same type as "var".
*     Should be from a Variable().
* @li lr: A scalar. Has the same type as "var".
* @li rho: A scalar. Has the same type as "var".
* @li momentum: A tensor. Has the same type as "var".
* @li epsilon: A scalar. Has the same type as "var".
* @li grad: A tensor for the gradient. Has the same type as "var".
* @li indices: A Tensor. Must be one of the following types: int32, int64.
*    A vector of indices into the first dimension of var, ms and mom.
*
* @par Attributes:
* use_locking: An optional bool. Defaults to "False".
*     If "True", updating of the "var", "ms", and "mom" tensors is protected
*     by a lock; otherwise the behavior is undefined, but may exhibit less
*     contention.
*
* @par Outputs:
* @li var: A mutable Tensor. Has the same type as "var". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator SparseApplyCenteredRMSProp.
*
*/
REG_CUST_OP(SparseApplyCenteredRMSProp)
  .INPUT(var, TensorType::RealNumberType())
  .INPUT(mg, TensorType::RealNumberType())
  .INPUT(ms, TensorType::RealNumberType())
  .INPUT(mom, TensorType::RealNumberType())
  .INPUT(lr, TensorType::RealNumberType())
  .INPUT(rho, TensorType::RealNumberType())
  .INPUT(momentum, TensorType::RealNumberType())
  .INPUT(epsilon, TensorType::RealNumberType())
  .INPUT(grad, TensorType::RealNumberType())
  .INPUT(indices, TensorType::RealIndexNumberType())
  .OUTPUT(var, TensorType::RealNumberType())
  .ATTR(use_locking, Bool, false)
  .CUST_OP_END_FACTORY_REG(SparseApplyCenteredRMSProp)
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H