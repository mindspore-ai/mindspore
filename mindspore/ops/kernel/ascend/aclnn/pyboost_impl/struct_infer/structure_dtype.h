/**
 * Copyright 2026 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_STRUCT_INFER_STRUCTURE_DTYPE_H_
#define MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_STRUCT_INFER_STRUCTURE_DTYPE_H_

#include <initializer_list>
#include <vector>

#include "ir/dtype.h"
#include "ir/tensor.h"

namespace mindspore::kernel::pyboost::op_infer {
bool IsIntegralType(TypeId t, bool include_bool);
bool IsFloatingType(TypeId t);
TypeId PromoteTypes(TypeId a, TypeId b);
TypeId ResultType(const tensor::TensorPtr &lhs, const tensor::TensorPtr &rhs);
TypeId ResultType(std::initializer_list<tensor::TensorPtr> tensors);
TypeId ResultType(const std::vector<tensor::TensorPtr> &tensors);
}  // namespace mindspore::kernel::pyboost::op_infer

#endif  // MINDSPORE_MINDSPORE_OPS_KERNEL_ASCEND_ACLNN_PYBOOST_IMPL_STRUCT_INFER_STRUCTURE_DTYPE_H_
