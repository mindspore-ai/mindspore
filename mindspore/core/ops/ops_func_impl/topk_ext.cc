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

#include "ops/ops_func_impl/topk_ext.h"
#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include "ops/op_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
TypePtr TopkExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<abstract::AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto output0_type = input_args[kInputIndex0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", output0_type, common_valid_types, prim_name);
  auto k_type = input_args[kInputIndex1]->GetType();
  const std::set<TypePtr> int_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTypeValid("k", k_type, int_types, prim_name);
  auto output1_type = kInt64;
  return std::make_shared<Tuple>(std::vector<TypePtr>{output0_type, output1_type});
}

BaseShapePtr TopkExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<abstract::AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShape());
  auto x_shape = shape_map[kShape];
  if (IsDynamicRank(x_shape)) {
    abstract::BaseShapePtr out_shape_ptr =
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape_ptr, out_shape_ptr});
  }

  if ((IsDynamicRank(x_shape)) || !IsValueKnown(input_args[kInputIndex1])) {
    auto unknown_shape_p = std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{unknown_shape_p, unknown_shape_p});
  }

  int64_t k_v = 0;
  // 2rd input is a Tensor when TopK is a dynamic shape operator
  if (CheckAndConvertUtils::IsTensor(input_args[kInputIndex1])) {
    auto k_dim = input_args[kInputIndex1]->GetShape()->GetShapeVector().size();
    if (k_dim > 1) {
      MS_LOG(EXCEPTION) << "For '" << prim_name
                        << "', the dimension of 'k' should only be 0 or 1 when 'k' is a Tensor, but got: " << k_dim
                        << ".";
    }
    auto k_val = CheckAndConvertUtils::CheckTensorIntValue("k", input_args[kInputIndex1]->GetValue(), prim_name,
                                                           input_args[kInputIndex1]->GetType());
    k_v = k_val[0];
  } else if (CheckAndConvertUtils::IsScalar(input_args[kInputIndex1])) {
    k_v = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue()).value();
  } else {
    MS_LOG(EXCEPTION) << "Invalid abstract type:" << input_args[kInputIndex1]->type_name();
  }

  if (!x_shape.empty()) {
    auto ndims = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue()).value();
    if (ndims < 0) {
      ndims = SizeToLong(x_shape.size()) + ndims;
    }
    if (x_shape[ndims] != abstract::Shape::kShapeDimAny) {
      std::pair<int64_t, int64_t> k_range(0, x_shape[ndims]);
      CheckAndConvertUtils::CheckInRange<int64_t>("k", k_v, kIncludeRight, k_range, prim_name);
      x_shape[ndims] = k_v;
    }
  }

  auto out_shape_ptr = std::make_shared<abstract::Shape>(x_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{out_shape_ptr, out_shape_ptr});
}

}  // namespace ops
}  // namespace mindspore
