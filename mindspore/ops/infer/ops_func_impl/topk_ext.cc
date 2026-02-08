/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/topk_ext.h"
#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "primitive/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace ops {
ShapeArray TopkExtFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &prim_name = primitive->name();
  auto &x = input_infos[kInputIndex0];
  auto k_opt = input_infos[kInputIndex1]->GetScalarValue<int64_t>();
  if (x->IsDynamicRank() || !k_opt.has_value()) {
    return {{abstract::Shape::kShapeRankAny}, {abstract::Shape::kShapeRankAny}};
  }
  auto k = k_opt.value();
  auto x_shape = x->GetShape();
  if (!CheckAndConvertUtils::IsEmptyTensorShape(x_shape) && !x_shape.empty()) {
    auto n_dims_opt = input_infos[kInputIndex2]->GetScalarValue<int64_t>();
    if (MS_UNLIKELY(!n_dims_opt.has_value())) {
      MS_EXCEPTION(ValueError) << "Failed to get 'n_dims' value.";
    }
    auto n_dims = n_dims_opt.value();
    CheckAndConvertUtils::CheckInRange<int64_t>("dim", n_dims, kIncludeLeft, {-x_shape.size(), x_shape.size()},
                                                prim_name);
    if (n_dims < 0) {
      n_dims = SizeToLong(x_shape.size()) + n_dims;
    }

    if (x_shape[n_dims] != abstract::Shape::kShapeDimAny) {
      std::pair<int64_t, int64_t> k_range(0, x_shape[n_dims]);
      CheckAndConvertUtils::CheckInRange<int64_t>("k", k, kIncludeBoth, k_range, prim_name);
      x_shape[n_dims] = k;
    }
  }
  return {x_shape, x_shape};
}

std::vector<TypeId> TopkExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  const auto x_type = input_infos[kInputIndex0]->GetType();
  const auto &prim_name = primitive->name();
  CheckAndConvertUtils::CheckTypeIdValid("input_x", x_type, common_valid_type_ids, prim_name);
  const auto k_type = input_infos[kInputIndex1]->GetType();
  CheckAndConvertUtils::CheckTypeIdValid("k", k_type, {kNumberTypeInt32, kNumberTypeInt64}, prim_name);
  return {x_type, kNumberTypeInt64};
}

REGISTER_SIMPLE_INFER(kNameTopkExt, TopkExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
