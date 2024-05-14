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
#include <map>
#include <set>
#include <string>
#include "ops/ops_func_impl/cross_ext.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {

void CheckCrossDim(const std::string &prim_name, const ShapeVector &input_shape, const AbstractBasePtr &input_dim) {
  if (input_dim->GetType()->isa<TypeNone>()) {
    int64_t dim_size_value = 3;
    for (size_t i = 0; i < input_shape.size(); i++) {
      if (input_shape[i] == dim_size_value) {
        break;
      }
      if (i == input_shape.size() - 1 && input_shape[i] != dim_size_value) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the size of inputs dim must be 3, but got "
                                 << input_shape[i] << ".";
      }
    }
  } else {
    auto dim_imm = GetScalarValue<int64_t>(input_dim->GetValue()).value();
    if (dim_imm < -static_cast<int64_t>(input_shape.size()) || dim_imm > static_cast<int64_t>(input_shape.size()) - 1) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', dim must be between "
                               << -static_cast<int64_t>(input_shape.size()) << " and "
                               << static_cast<int64_t>(input_shape.size()) - 1 << " , but got " << dim_imm << ".";
    }
  }
}

BaseShapePtr CrossExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShape())[kShape];
  auto other_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShape())[kShape];
  // support dynamic rank
  if (IsDynamicRank(input_shape) || IsDynamicRank(other_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  // Support Dynamic Shape
  if (IsDynamic(input_shape) || IsDynamic(other_shape)) {
    ShapeVector shape_out;
    for (size_t i = 0; i < input_shape.size(); ++i) {
      shape_out.push_back(abstract::Shape::kShapeDimAny);
    }
    return std::make_shared<abstract::Shape>(shape_out);
  }
  const std::map<std::string, BaseShapePtr> shapes = {{"input", input_args[0]->GetShape()},
                                                      {"other", input_args[1]->GetShape()}};
  std::vector<int64_t> valid_shape;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    valid_shape.push_back(SizeToLong(input_shape[i]));
  }
  (void)CheckAndConvertUtils::CheckTensorShapeSame(shapes, valid_shape, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("dim of input", SizeToLong(input_shape.size()), kGreaterThan, 0, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("dim of input", SizeToLong(other_shape.size()), kGreaterThan, 0, prim_name);
  CheckCrossDim(prim_name, input_shape, input_args[kInputIndex2]);
  return std::make_shared<abstract::Shape>(input_shape);
}

TypePtr CrossExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  const std::set<TypePtr> valid_types = {kInt8,    kInt16,   kInt32,   kInt64,     kUInt8,
                                         kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  MS_EXCEPTION_IF_NULL(input_args[kIndex0]);
  MS_EXCEPTION_IF_NULL(input_args[kIndex1]);
  auto input_type = input_args[kIndex0]->GetType();
  auto other_type = input_args[kIndex1]->GetType();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input", input_type);
  (void)types.emplace("other", other_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
  return input_type;
}
}  // namespace ops
}  // namespace mindspore
