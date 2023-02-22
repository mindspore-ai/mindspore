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

#include "ops/csr_reducesum.h"

#include <memory>

#include "abstract/dshape.h"
#include "abstract/param_validator.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "abstract/abstract_value.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/scalar.h"
#include "mindapi/base/shape_vector.h"
#include "ops/core_ops.h"
#include "ops/primitive_c.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
using abstract::AbstractScalar;
using abstract::AbstractTensor;
using abstract::AbstractTuple;
AbstractBasePtr CSRReduceSumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  // Inputs: a sparse tensor and an axis.
  constexpr auto kCSRReduceSumInputsNum = 5;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, input_args, kCSRReduceSumInputsNum);
  auto indptr = abstract::CheckArg<AbstractTensor>(op_name, input_args, 0);
  auto indices = abstract::CheckArg<AbstractTensor>(op_name, input_args, 1);
  auto values = abstract::CheckArg<AbstractTensor>(op_name, input_args, 2);
  auto shape = abstract::CheckArg<AbstractTuple>(op_name, input_args, 3);
  auto axis = abstract::CheckArg<AbstractScalar>(op_name, input_args, 4);
  MS_EXCEPTION_IF_NULL(indptr);
  MS_EXCEPTION_IF_NULL(indices);
  MS_EXCEPTION_IF_NULL(values);
  MS_EXCEPTION_IF_NULL(shape);
  MS_EXCEPTION_IF_NULL(axis);

  CheckSparseIndicesDtypeInt32(indptr->element()->BuildType(), "Indptr");
  CheckSparseIndicesDtypeInt32(indices->element()->BuildType(), "Indices");

  ShapeVector sparse_shape = ConvertToShapeVector(shape);
  ShapeVector out_shape = sparse_shape;
  MS_EXCEPTION_IF_NULL(axis->BuildValue());
  if (axis->BuildValue()->isa<Int32Imm>() || axis->BuildValue()->isa<Int64Imm>()) {
    int64_t axis_value = GetValue<int64_t>(axis->BuildValue());
    int64_t dim = static_cast<int64_t>(sparse_shape.size());
    if (axis_value != 1 && axis_value != 1 - dim) {
      MS_EXCEPTION(ValueError) << "For CSRReduceSum, `axis` should be 1 or 1-dim. But got `axis`: " << axis_value
                               << "and `1- dim`: " << 1 - dim << ".";
    }
    if (axis_value < 0) {
      axis_value += dim;
    }
    out_shape[LongToSize(axis_value)] = 1;
    primitive->set_attr(kCSRAxis, MakeValue(axis_value));
  } else {
    MS_EXCEPTION(TypeError) << "For CSRReduceSum, `axis` should be int32 or int64, but got "
                            << axis->BuildType()->ToString() << ".";
  }

  MS_EXCEPTION_IF_NULL(values->element());
  auto ret = std::make_shared<AbstractTensor>(values->element()->BuildType(), out_shape);
  // SetAttr
  auto nnz_vec = indices->shape()->shape();
  auto csr_avg_rows = nnz_vec[0] / sparse_shape[0];
  primitive->set_attr(kCSRAvgRows, MakeValue(csr_avg_rows));
  primitive->set_attr(kIsCSR, MakeValue(true));
  return ret;
}
MIND_API_OPERATOR_IMPL(CSRReduceSum, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(CSRReduceSum, prim::kPrimCSRReduceSum, CSRReduceSumInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
