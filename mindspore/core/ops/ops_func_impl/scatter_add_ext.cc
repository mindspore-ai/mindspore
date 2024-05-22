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

#include "ops/ops_func_impl/scatter_add_ext.h"
#include <map>
#include <string>
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ScatterAddExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto input_shape = input_args[kIndex0]->GetShape();
  auto index_shape = input_args[kIndex2]->GetShape();
  auto src_shape = input_args[kIndex3]->GetShape();
  if (input_shape->IsDynamic() || index_shape->IsDynamic() || src_shape->IsDynamic()) {
    return input_shape->cast<abstract::ShapePtr>();
  }
  auto dim_opt = GetScalarValue<int64_t>(input_args[kIndex1]->GetValue());
  if (MS_UNLIKELY(!dim_opt.has_value())) {
    return input_shape->Clone();
  }
  auto dim = dim_opt.value();
  auto input_shape_vec = input_shape->GetShapeVector();
  auto index_shape_vec = index_shape->GetShapeVector();
  auto src_shape_vec = src_shape->GetShapeVector();
  auto input_rank = SizeToLong(input_shape_vec.size());
  MS_CHECK_VALUE(
    dim >= -input_rank && dim < input_rank,
    CheckAndConvertUtils::FormatCheckInRangeMsg("dim", dim, kIncludeLeft, {-input_rank, input_rank}, primitive));
  if (input_shape_vec.size() < 1 || index_shape_vec.size() < 1 || src_shape_vec.size() < 1) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", dimension size of 'input', 'index' and "
                             << "'src' must be greater than or equal to 1. But got input_shape: " << input_shape_vec
                             << ", index_shape: " << index_shape_vec << ", src_shape: " << src_shape_vec << ".";
  }
  if (input_shape_vec.size() != index_shape_vec.size() || input_shape_vec.size() != src_shape_vec.size()) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the dimension of 'input', 'index' and 'src' should be same, but got "
                             << "'input' dims: " << input_shape_vec.size() << "; "
                             << "'index' dims: " << index_shape_vec.size() << "; "
                             << "'src' dims: " << src_shape_vec.size() << ".";
  }
  auto final_dim = dim >= 0 ? dim : dim + input_rank;
  for (int64_t d = 0; d < input_rank; d++) {
    if (d != final_dim && index_shape_vec[d] > input_shape_vec[d]) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", "
                               << "except for the dimension specified by 'dim', the size of each dimension of 'index' "
                                  "must be less than or equal to that of 'input'. But got "
                               << d << "th dim of 'index' and 'input' " << index_shape_vec[d] << ", "
                               << input_shape_vec[d] << "respectively.";
    }
  }
  return input_shape->cast<abstract::ShapePtr>();
}

TypePtr ScatterAddExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kIndex0]->GetType();
  auto src_type = input_args[kIndex3]->GetType();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input", input_type);
  (void)types.emplace("src", src_type);
  (void)CheckAndConvertUtils::CheckTypeSame(types, primitive->name());
  const std::set<TypePtr> valid_types = {kInt8,    kInt16,   kInt32,   kInt64, kUInt8,
                                         kFloat16, kFloat32, kFloat64, kBool,  kBFloat16};
  (void)CheckAndConvertUtils::CheckTypeValid("input", input_type, valid_types, primitive->name());
  return input_args[kIndex0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
