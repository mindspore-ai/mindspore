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

#include "ops/csr_gather.h"

#include <memory>

#include "abstract/dshape.h"
#include "abstract/param_validator.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/anf_utils.h"
#include "abstract/abstract_value.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
using abstract::AbstractTensor;
using abstract::AbstractTuple;
AbstractBasePtr CSRGatherInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  // Inputs: the indptr and indices of a sparse csr tensor, a dense tensor, and the shape of the sparse tensor.
  constexpr size_t csr_row_num = 2;
  const std::string op_name = primitive->name();
  abstract::CheckArgsSize(op_name, input_args, kSizeFour);
  auto indptr = abstract::CheckArg<AbstractTensor>(op_name, input_args, kIndexZero);
  auto indices = abstract::CheckArg<AbstractTensor>(op_name, input_args, kIndexOne);
  auto dense = abstract::CheckArg<AbstractTensor>(op_name, input_args, kIndexTwo);
  auto sparse_shape = abstract::CheckArg<AbstractTuple>(op_name, input_args, kIndexThree);
  MS_EXCEPTION_IF_NULL(indptr);
  MS_EXCEPTION_IF_NULL(indices);
  MS_EXCEPTION_IF_NULL(dense);
  MS_EXCEPTION_IF_NULL(sparse_shape);

  CheckSparseIndicesDtypeInt32(indptr->element()->BuildType(), "Indptr");
  CheckSparseIndicesDtypeInt32(indices->element()->BuildType(), "Indices");

  auto shape_value = sparse_shape->BuildValue()->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(shape_value);
  auto nnz_vec = indices->shape()->shape();
  int64_t csr_avg_rows = nnz_vec[0] / GetValue<int64_t>(shape_value->value()[0]);
  primitive->set_attr(kCSRAvgRows, MakeValue(csr_avg_rows));
  primitive->set_attr(kIsCSR, MakeValue(true));

  MS_EXCEPTION_IF_NULL(indices->shape());
  ShapeVector out_shape = indices->shape()->shape();
  MS_EXCEPTION_IF_NULL(dense->shape());
  ShapeVector dense_shape = dense->shape()->shape();
  for (size_t i = csr_row_num; i < dense_shape.size(); ++i) {
    out_shape.push_back(dense_shape[i]);
  }
  MS_EXCEPTION_IF_NULL(dense->element());
  auto ret = std::make_shared<AbstractTensor>(dense->element()->BuildType(), out_shape);
  return ret;
}
MIND_API_OPERATOR_IMPL(CSRGather, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(CSRGather, prim::kPrimCSRGather, CSRGatherInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
