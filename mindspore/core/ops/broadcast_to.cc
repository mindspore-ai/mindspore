/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr BroadcastToInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto value_ptr = primitive->GetAttr(kShape);
  auto input_x = GetValue<std::vector<int64_t>>(value_ptr);
  CheckAndConvertUtils::Check("x shape", SizeToLong(x_shape.size()), kLessEqual, SizeToLong(input_x.size()), prim_name);
  if (!x_shape.empty() && x_shape[0] == -2) {
    auto x_shape_ptr = std::make_shared<abstract::Shape>(input_x);
    return x_shape_ptr;
  }
  auto outer_dim_offset = input_x.size() - x_shape.size();
  bool flag = true;
  if (input_x.end() == find(input_x.begin(), input_x.end(), -1)) {
    flag = false;
  } else {
    flag = true;
  }

  if (flag) {
    for (size_t i = 0; i < input_x.size(); i++) {
      if (input_x[i] == -1) {
        if (i < outer_dim_offset) {
          MS_EXCEPTION(ValueError) << "For '" << prim_name
                                   << "', -1 in init shape is in an incompatible "
                                      "location with given input tensor, -1 index in init shape: "
                                   << i << " but -1 can only be in index" << x_shape.size()
                                   << "onwards for this input.";
        }
        input_x[i] = x_shape[i - outer_dim_offset];
      }
    }
  }
  auto x_shape_ptr = std::make_shared<abstract::Shape>(input_x);
  (void)primitive->AddAttr("shape", MakeValue(input_x));
  for (size_t i = 0; i < x_shape.size(); i++) {
    if (x_shape[i] == -1) {
      continue;
    }
    if (input_x[i + outer_dim_offset] != x_shape[i] && x_shape[i] != 1) {
      MS_EXCEPTION(ValueError)
        << "For '" << prim_name
        << "', in order to broadcast, each dimension pair must be equal or input dimension is 1 or target "
           "dimension is -1. But got x_shape: "
        << input_args[0]->BuildShape()->ToString() << ", target shape: " << x_shape_ptr->ToString() << ".";
    }
  }
  return x_shape_ptr;
}

TypePtr BroadcastToInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_dtype = input_args[0]->BuildType()->cast<TensorTypePtr>();
  std::set<TypePtr> template_types = {kTensorType};
  (void)CheckAndConvertUtils::CheckSubClass("x_dtype", x_dtype, template_types, prim->name());
  return x_dtype->element();
}
}  // namespace

MIND_API_OPERATOR_IMPL(BroadcastTo, BaseOperator);
void BroadcastTo::Init(const std::vector<int64_t> &shape) { set_shape(shape); }

void BroadcastTo::set_shape(const std::vector<int64_t> &shape) {
  (void)CheckAndConvertUtils::CheckInteger(kShapeSize, SizeToLong(shape.size()), kGreaterThan, 0, name());
  (void)AddAttr(kShape, api::MakeValue(shape));
}

std::vector<int64_t> BroadcastTo::get_shape() const {
  auto value_ptr = GetAttr(kShape);
  return GetValue<std::vector<int64_t>>(value_ptr);
}
AbstractBasePtr BroadcastToInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(BroadcastToInferShape(primitive, input_args),
                                BroadcastToInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(BroadcastTo, prim::kPrimBroadcastTo, BroadcastToInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
