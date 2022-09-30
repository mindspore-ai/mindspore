/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#include <set>
#include <string>
#include "ops/cumsum.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_NAME_IMPL(CumSum, kNameCumSum, BaseOperator);
void CumSum::Init(const bool exclusive, const bool reverse) {
  this->set_exclusive(exclusive);
  this->set_reverse(reverse);
}

void CumSum::set_exclusive(const bool exclusive) { (void)this->AddAttr(kExclusive, api::MakeValue(exclusive)); }

bool CumSum::get_exclusive() const {
  auto value_ptr = this->GetAttr(kExclusive);
  return GetValue<bool>(value_ptr);
}

void CumSum::set_reverse(const bool reverse) { (void)this->AddAttr(kReverse, api::MakeValue(reverse)); }

bool CumSum::get_reverse() const {
  auto value_ptr = this->GetAttr(kReverse);
  return GetValue<bool>(value_ptr);
}

void CumSum::set_axis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

int64_t CumSum::get_axis() const {
  auto value_ptr = this->GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}

abstract::ShapePtr CumSumInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape_ptr = input_args[kInputIndex0]->BuildShape();
  if (x_shape_ptr->IsDynamic()) {
    return x_shape_ptr->cast<abstract::ShapePtr>();
  }
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x_shape_ptr)[kShape];
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
  }
  auto rank = SizeToLong(x_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("rank of 'x'", rank, kGreaterThan, 0, prim_name);

  int64_t axis;
  if (input_args[kInputIndex1]->isa<abstract::AbstractTensor>()) {
    auto axis_ptr = input_args[kInputIndex1]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(axis_ptr);
    auto axis_value_ptr = axis_ptr->BuildValue();
    MS_EXCEPTION_IF_NULL(axis_value_ptr);
    if (axis_value_ptr->isa<tensor::Tensor>()) {
      auto axis_tensor = axis_value_ptr->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(axis_tensor);
      if (axis_tensor->data_type_c() == TypeId::kNumberTypeInt64) {
        axis = *static_cast<int64_t *>(axis_tensor->data_c());
      } else if (axis_tensor->data_type_c() == TypeId::kNumberTypeInt32) {
        axis = *static_cast<int32_t *>(axis_tensor->data_c());
      } else {
        MS_LOG(EXCEPTION) << "For '" << primitive->name()
                          << "', the second input type should be tensor with type int64 or int32, but got tensor type:"
                          << TypeIdToString(axis_tensor->data_type());
      }
    } else {
      return std::make_shared<abstract::Shape>(x_shape);
    }
  } else if (input_args[kInputIndex1]->isa<abstract::AbstractScalar>()) {
    auto axis_ptr = input_args[kInputIndex1]->cast<abstract::AbstractScalarPtr>();
    MS_EXCEPTION_IF_NULL(axis_ptr);
    axis = GetValue<int64_t>(axis_ptr->BuildValue());
  } else {
    MS_LOG(EXCEPTION) << "For '" << primitive->name()
                      << "', the second input type should be tensor or scalar, but got invalid abstract type:"
                      << input_args[kInputIndex1]->type_name() << ".";
  }
  CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis, kIncludeBoth, {-rank, rank - 1}, prim_name);
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr CumSumInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool is_ascend = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  // First input.
  std::set<TypePtr> valid_x_types;
  if (is_ascend) {
    valid_x_types = {kInt8, kUInt8, kInt32, kFloat16, kFloat32, kFloat64};
  } else {
    valid_x_types = common_valid_types_with_complex;
  }
  auto x_type = input_args[kInputIndex0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_x_types, prim_name);
  // Second input.
  auto axis_type = input_args[kInputIndex1]->BuildType();
  const std::set<TypePtr> valid_axis_types = {kInt64};
  (void)CheckAndConvertUtils::CheckTypeValid("axis", axis_type, valid_axis_types, prim_name);
  return x_type;
}

abstract::AbstractBasePtr CumSumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = CumSumInferType(primitive, input_args);
  auto shapes = CumSumInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}
REGISTER_PRIMITIVE_EVAL_IMPL(CumSum, prim::kPrimCumSum, CumSumInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
