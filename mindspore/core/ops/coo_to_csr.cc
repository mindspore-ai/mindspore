/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ops/coo_to_csr.h"

#include <memory>

#include "abstract/param_validator.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
using abstract::AbstractScalar;
using abstract::AbstractTensor;
using abstract::AbstractTuple;
AbstractBasePtr COO2CSRInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  // Inputs: the row indices of a sparse coo tensor, and the size of its first dimension.
  constexpr auto kCSRArgsSize = 2;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, input_args, kCSRArgsSize);
  auto row_indices = abstract::CheckArg<AbstractTensor>(op_name, input_args, 0);
  auto height = abstract::CheckArg<AbstractScalar>(op_name, input_args, 1);
  MS_EXCEPTION_IF_NULL(row_indices);
  MS_EXCEPTION_IF_NULL(height);
  CheckSparseIndicesDtypeInt32(row_indices->element()->BuildType(), "row_indices");
  MS_EXCEPTION_IF_NULL(height->BuildValue());
  ShapeVector out_shape;
  if (height->BuildValue()->isa<Int32Imm>() || height->BuildValue()->isa<Int64Imm>()) {
    int64_t height_value = GetValue<int64_t>(height->BuildValue());
    out_shape.push_back(height_value + 1);
  } else {
    MS_EXCEPTION(ValueError) << "Currently, only support Integer height.";
  }

  MS_EXCEPTION_IF_NULL(row_indices->element());
  auto ret = std::make_shared<AbstractTensor>(row_indices->element()->BuildType(), out_shape);
  return ret;
}
MIND_API_OPERATOR_IMPL(COO2CSR, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(COO2CSR, prim::kPrimCOO2CSR, COO2CSRInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
