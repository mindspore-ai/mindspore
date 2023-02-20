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

#include "ops/csr_elementwise.h"

#include <memory>

#include "abstract/dshape.h"
#include "abstract/param_validator.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
using abstract::AbstractTensor;
using abstract::AbstractTuple;
AbstractBasePtr CSRElementWiseInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  // Inputs: a sparse tensor and a dense tensor.
  constexpr auto kCSRElementwiseInputsNum = 5;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, input_args, kCSRElementwiseInputsNum);
  auto indptr = abstract::CheckArg<AbstractTensor>(op_name, input_args, 0);
  auto indices = abstract::CheckArg<AbstractTensor>(op_name, input_args, 1);
  auto values = abstract::CheckArg<AbstractTensor>(op_name, input_args, 2);
  auto shape = abstract::CheckArg<AbstractTuple>(op_name, input_args, 3);
  auto dense = abstract::CheckArg<AbstractTensor>(op_name, input_args, 4);
  MS_EXCEPTION_IF_NULL(indptr);
  MS_EXCEPTION_IF_NULL(indices);
  MS_EXCEPTION_IF_NULL(values);
  MS_EXCEPTION_IF_NULL(shape);
  MS_EXCEPTION_IF_NULL(dense);

  CheckSparseIndicesDtypeInt32(indptr->element()->BuildType(), "Indptr");
  CheckSparseIndicesDtypeInt32(indices->element()->BuildType(), "Indices");

  ShapeVector sparse_shape = ConvertToShapeVector(shape);
  auto dense_shape = dense->shape()->shape();
  CheckSparseShape(sparse_shape, dense_shape);
  auto ret = values->Broaden();
  // SetAttr
  auto nnz_vec = indices->shape()->shape();
  auto csr_avg_rows = nnz_vec[0] / dense_shape[0];
  primitive->set_attr(kCSRAvgRows, MakeValue(csr_avg_rows));
  primitive->set_attr(kIsCSR, MakeValue(true));
  return ret;
}
MIND_API_OPERATOR_IMPL(CSRMul, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(CSRMul, prim::kPrimCSRMul, CSRElementWiseInfer, nullptr, true);
MIND_API_OPERATOR_IMPL(CSRDiv, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(CSRDiv, prim::kPrimCSRDiv, CSRElementWiseInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
