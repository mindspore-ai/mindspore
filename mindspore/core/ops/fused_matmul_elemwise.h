/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_FUSED_MATMUL_ELEMWISE_H_
#define MINDSPORE_CORE_OPS_FUSED_MATMUL_ELEMWISE_H_
#include <memory>
#include <string>
#include <vector>
#include "mindapi/base/types.h"
#include "utils/ms_context.h"
#include "ops/base_operator.h"
#include "ops/ops_func_impl/matmul.h"

namespace mindspore {
namespace ops {
/// \brief Multiplies matrix a and matrix b. Refer to Python API @ref mindspore.ops.FusedMatMulElemBinary for more
/// details.
class MIND_API FusedMatMulElemBinary : public MatMulFuncImpl {
 public:
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
};

/// \brief Multiplies matrix a and matrix b. Refer to Python API @ref mindspore.ops.FusedMatMulElemUnary for more
/// details.
class MIND_API FusedMatMulElemUnary : public MatMulFuncImpl {
 public:
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override;
};

}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_FUSED_MATMUL_ELEMWISE_H_
