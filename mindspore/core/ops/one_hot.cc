/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include <string>
#include "ops/one_hot.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void OneHot::Init(const int64_t axis) { this->set_axis(axis); }
void OneHot::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, MakeValue(axis)); }

int64_t OneHot::get_axis() const { return GetValue<int64_t>(GetAttr(kAxis)); }
namespace {
abstract::ShapePtr OneHotInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  int64_t axis = GetValue<int64_t>(primitive->GetAttr(kAxis));
  const size_t depth_index = 1;
  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
  auto in_shape = shape_map[kShape];
  auto min_shape = shape_map[kMinShape];
  auto max_shape = shape_map[kMaxShape];
  CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis, kIncludeBoth, {-1, SizeToLong(in_shape.size())}, op_name);
  auto depth = input_args[depth_index]->BuildValue();
  MS_EXCEPTION_IF_NULL(depth);
  int64_t depth_value;
  if (depth->isa<tensor::Tensor>()) {
    (void)CheckAndConvertUtils::CheckTensorTypeValid("depth", input_args[1]->BuildType(), {kInt64}, op_name);
    auto depth_data = depth->cast<tensor::TensorPtr>()->data_c();
    MS_EXCEPTION_IF_NULL(depth_data);
    auto data_value = reinterpret_cast<int64_t *>(depth_data);
    depth_value = *data_value;
  } else if (depth->isa<Int64Imm>()) {
    depth_value = GetValue<int64_t>(depth);
  } else {
    MS_EXCEPTION(TypeError) << "OneHot depth must be a tensor or number of int64.";
  }

  (void)CheckAndConvertUtils::CheckInteger("depth value", depth_value, kGreaterEqual, 0, op_name);
  if (min_shape.size() == 0 || max_shape.size() == 0) {
    if (axis >= 0) {
      (void)in_shape.insert(in_shape.begin() + axis, depth_value);
    } else {
      in_shape.push_back(depth_value);
    }
  } else {
    if (axis >= 0) {
      (void)in_shape.insert(in_shape.begin() + axis, depth_value);
      (void)min_shape.insert(min_shape.begin() + axis, depth_value);
      (void)max_shape.insert(max_shape.begin() + axis, depth_value);
    } else {
      in_shape.push_back(depth_value);
      min_shape.push_back(depth_value);
      max_shape.push_back(depth_value);
    }
  }
  return std::make_shared<abstract::Shape>(in_shape, min_shape, max_shape);
}

TypePtr OneHotInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", input_args[kInputIndex0]->BuildType(), {kInt32, kInt64},
                                                   op_name);
  (void)CheckAndConvertUtils::CheckTypeValid("depth", input_args[kInputIndex1]->BuildType(),
                                             {kInt8, kInt16, kInt32, kInt64}, op_name);
  std::map<std::string, TypePtr> args = {{"on_value", input_args[kInputIndex2]->BuildType()},
                                         {"off_dtype", input_args[kInputIndex3]->BuildType()}};
  return CheckAndConvertUtils::CheckTensorTypeSame(args, {kFloat16, kFloat32}, op_name);
}
}  // namespace
AbstractBasePtr OneHotInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = OneHotInferType(primitive, input_args);
  auto infer_shape = OneHotInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(OneHot, prim::kPrimOneHot, OneHotInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
