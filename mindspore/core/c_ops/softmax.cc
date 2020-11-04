/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "c_ops/softmax.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "c_ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
void Softmax::set_axis(const std::vector<int> &axis) { this->set_attr(kAxis, MakeValue(axis)); }

std::vector<int> Softmax::get_axis() const {
  auto value_ptr = GetAttr(kAxis);
  return GetValue<std::vector<int>>(value_ptr);
}

void Softmax::Init(int axis) {
  auto op_name = this->name();
  std::vector<int> axis_vec = {axis};
  CheckAndConvertUtils::CheckInteger("axis_len", axis_vec.size(), kEqual, 1, op_name);
  auto rank = axis_vec.size();
  for (auto &item : axis_vec) {
    CheckAndConvertUtils::CheckInRange("axis", item, kIncludeLeft, {-rank, rank}, op_name);
  }
  this->set_axis(axis_vec);
}

abstract::ShapePtr InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto softmax_prim = primitive->cast<PrimSoftmaxPtr>();
  MS_EXCEPTION_IF_NULL(softmax_prim);
  auto op_name = softmax_prim->name();
  auto axis = softmax_prim->get_axis();
  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShape("input_shape", input_args[0]->GetShapeTrack(), op_name);
  auto rank = in_shape.size();
  for (auto &item : axis) {
    CheckAndConvertUtils::CheckInRange("axis", item, kIncludeLeft, {-rank, rank}, op_name);
  }
  return std::make_shared<abstract::Shape>(in_shape);
}

TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  if (std::any_of(input_args.begin(), input_args.end(), [](const AbstractBasePtr &a) { return a == nullptr; })) {
    MS_LOG(EXCEPTION) << "nullptr";
  }
  std::map<std::string, TypePtr> types;
  const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64};
  types.emplace("x", input_args[0]->BuildType());
  auto infer_type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return TypeIdToType(infer_type);
}

AbstractBasePtr SoftmaxInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(InferType(primitive, input_args),
                                                    InferShape(primitive, input_args)->shape());
}

REGISTER_PRIMITIVE_EVAL_IMPL(Softmax, prim::kPrimSoftmax, SoftmaxInfer);
}  // namespace mindspore
