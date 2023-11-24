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

#include <memory>
#include <vector>
#include <algorithm>
#include "ir/anf.h"
#include "util/log_adapter.h"
#include "utils/shape_utils.h"
#include "abstract/dshape.h"
#include "ops/op_utils.h"
#include "ops/op_name.h"
#include "ops/ops_func_impl/qr.h"

namespace mindspore::ops {
BaseShapePtr QrFuncImpl::InferShape(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]->GetShape());
  auto input_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  auto full_matrices_opt = GetScalarValue<bool>(input_args[kInputIndex1]->GetValue());
  if (IsDynamicRank(input_shape) || !full_matrices_opt.has_value()) {
    auto unknow_rank_q = std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    auto unknow_rank_r = std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{unknow_rank_q, unknow_rank_r});
  }
  const int64_t kDimLeastNum = 2;
  MS_CHECK_VALUE(input_shape.size() >= kDimLeastNum, "The input matrix must be at least two dimensions");

  std::vector<int64_t> out_q_dims(input_shape.begin(), input_shape.end());
  std::vector<int64_t> out_r_dims(input_shape.begin(), input_shape.end());
  const int64_t penultimate = 2;
  const int64_t last_dimension = input_shape.size() - 1;
  const int64_t penultimate_dimension = input_shape.size() - penultimate;
  if (full_matrices_opt.value()) {
    out_q_dims[last_dimension] = out_q_dims[penultimate_dimension];
  } else {
    // input [m, n]
    // if m or n is kShapeDimAny and full_matrices = false
    // then q.shape = [m, -1] r.shape = [-1, n]
    auto m = input_shape[last_dimension];
    auto n = input_shape[penultimate_dimension];
    if (IsDynamicShape({m, n})) {
      out_q_dims[last_dimension] = abstract::Shape::kShapeDimAny;
      out_r_dims[penultimate_dimension] = abstract::Shape::kShapeDimAny;
    } else {
      auto p = std::min(input_shape[penultimate_dimension], input_shape[last_dimension]);
      out_q_dims[last_dimension] = p;
      out_r_dims[penultimate_dimension] = p;
    }
  }

  auto q_shape = std::make_shared<abstract::TensorShape>(out_q_dims);
  auto r_shape = std::make_shared<abstract::TensorShape>(out_r_dims);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{q_shape, r_shape});
}

TypePtr QrFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kInputIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(input_type);
  return std::make_shared<Tuple>(std::vector<TypePtr>{input_type, input_type});
}
}  // namespace mindspore::ops
