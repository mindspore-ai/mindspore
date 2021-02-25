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

#include <set>
#include "ops/broadcast_to.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr BroadcastToInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto broad_cast_to = primitive->cast<PrimBroadcastToPtr>();
  MS_EXCEPTION_IF_NULL(broad_cast_to);
  auto prim_name = broad_cast_to->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShape("x_shape", input_args[0]->BuildShape(), prim_name);
  auto input_x = broad_cast_to->get_shape();
  int64_t outer_dim_offset = input_x.size() - x_shape.size();
  CheckAndConvertUtils::Check("x shape", x_shape, kLessEqual, "input_x", input_x, prim_name);
  bool flag = true;
  if (input_x.end() == find(input_x.begin(), input_x.end(), -1)) {
    flag = false;
  } else {
    flag = true;
  }
  if (flag == true) {
    for (int64_t i = 0; i < (int64_t)input_x.size(); i++) {
      if (input_x[i] == -1) {
        if (i < outer_dim_offset) {
          MS_EXCEPTION(ValueError) << " -1 in init shape is in an incompatible "
                                      "location with given input tensor, -1 index in init shape: "
                                   << i << " but -1 can only be in index" << x_shape.size()
                                   << "onwards for this input.";
        }
        input_x[i] = x_shape[i - outer_dim_offset];
      }
    }
  }
  std::reverse(input_x.begin(), input_x.end());
  return std::make_shared<abstract::Shape>(input_x);
}

TypePtr BroadcastToInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_dtype = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  std::set<TypePtr> template_types = {TypeIdToType(kObjectTypeTensorType)};
  CheckAndConvertUtils::CheckSubClass("x_dtype", x_dtype, template_types, prim->name());
  auto infer_dtype = input_args[0]->BuildType()->type_id();
  return TypeIdToType(infer_dtype);
}
}  // namespace

void BroadcastTo::Init(const std::vector<int64_t> &shape) { set_shape(shape); }

void BroadcastTo::set_shape(const std::vector<int64_t> &shape) {
  CheckAndConvertUtils::CheckInteger(kShapeSize, shape.size(), kGreaterThan, 0, name());
  AddAttr(kShape, MakeValue(shape));
}

std::vector<int64_t> BroadcastTo::get_shape() const {
  auto value_ptr = GetAttr(kShape);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
AbstractBasePtr BroadcastToInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  return std::make_shared<abstract::AbstractTensor>(BroadcastToInferType(primitive, input_args),
                                                    BroadcastToInferShape(primitive, input_args)->shape());
}
REGISTER_PRIMITIVE_C(kNameBroadcastTo, BroadcastTo);
}  // namespace ops
}  // namespace mindspore
