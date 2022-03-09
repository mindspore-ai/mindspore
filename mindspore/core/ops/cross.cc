/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/cross.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr CrossInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto x1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto x2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto dim = GetValue<int64_t>(primitive->GetAttr("dim"));
  if (x1_shape.size() != x2_shape.size()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', The shape of two inputs must have the same size.";
  }
  for (size_t i = 0; i < x1_shape.size(); ++i) {
    if (x1_shape[i] != x2_shape[i]) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', x1 and x2 must have the same shape.";
    }
  }
  if (x1_shape.size() <= 0 || x2_shape.size() <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', inputs should not be a " << x1_shape.size()
                             << " dimensional tensor.";
  }
  int64_t default_dim = -65530;
  if (dim == default_dim) {
    int64_t dim_size_value = 3;
    for (size_t i = 0; i < x1_shape.size(); i++) {
      if (x1_shape[i] == dim_size_value) {
        dim = i;
        break;
      }
      if (i == x1_shape.size() - 1 && x1_shape[i] != dim_size_value) {
        MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', The size of inputs dim should be 3, but got "
                                 << x1_shape[i];
      }
    }
  }
  if ((dim < -static_cast<int64_t>(x1_shape.size()) || dim > static_cast<int64_t>(x1_shape.size()) - 1) &&
      dim != default_dim) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "',dim should be between "
                             << -static_cast<int64_t>(x1_shape.size()) << " and "
                             << static_cast<int64_t>(x1_shape.size()) - 1 << " ,but got " << dim;
  }
  if (dim < 0 && dim != default_dim) {
    dim = static_cast<int64_t>(x1_shape.size()) + dim;
  }
  int64_t dim_size = 3;
  if (x1_shape[dim] != dim_size && x2_shape[dim] != dim_size && dim != default_dim) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', The size of inputs dim should be 3,but got "
                             << x1_shape[dim];
  }
  return std::make_shared<abstract::Shape>(x1_shape);
}

TypePtr CrossInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInteger("Cross infer", input_args.size(), kEqual, input_num, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kInt8,    kInt16,  kInt32,  kInt64,  kUInt8,     kFloat16,   kFloat32,
                                         kFloat64, kUInt16, kUInt32, kUInt64, kComplex64, kComplex128};
  auto x1_type = input_args[0]->BuildType();
  auto x2_type = input_args[1]->BuildType();
  auto tensor_type = x2_type->cast<TensorTypePtr>();
  auto element = tensor_type->element();
  CheckAndConvertUtils::CheckTensorTypeValid("x2", x2_type, valid_types, primitive->name());
  return CheckAndConvertUtils::CheckTensorTypeValid("x1", x1_type, {element}, primitive->name());
}
AbstractBasePtr CrossInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = CrossInferType(primitive, input_args);
  auto infer_shape = CrossInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(Cross, prim::kPrimCross, CrossInfer, nullptr, true);
}  // namespace
}  // namespace ops
}  // namespace mindspore
