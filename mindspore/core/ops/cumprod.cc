/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "ops/cumprod.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr CumProdInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{UNKNOWN_RANK});
  }
  const int64_t min_dim = 0;
  (void)CheckAndConvertUtils::CheckInteger("rank of input", SizeToLong(x_shape.size()), kGreaterThan, min_dim,
                                           prim_name);
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
  auto y_rank = SizeToLong(x_shape.size());
  CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis, kIncludeBoth, {-y_rank, y_rank - 1}, prim_name);
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr CumProdInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", int64_t(input_args.size()), kEqual, input_num, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_type = input_args[0]->BuildType();
  auto axis_type = input_args[1]->BuildType();
  (void)CheckAndConvertUtils::CheckTypeValid("axis", axis_type, {kInt}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, common_valid_types_with_complex, prim_name);
  return x_type;
}

ValuePtr CumProdInferValue(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  if (input_args.empty()) {
    return nullptr;
  }
  auto axis = input_args[kInputIndex1]->BuildValue();
  if (axis == nullptr) {
    MS_EXCEPTION(ValueError) << "For " << prim->name() << ", the 'axis' cannot be None, but got " << axis;
  }
  return nullptr;
}
}  // namespace

MIND_API_OPERATOR_IMPL(CumProd, BaseOperator);
void CumProd::Init(const bool exclusive, const bool reverse) {
  this->SetExclusive(exclusive);
  this->SetReverse(reverse);
}

void CumProd::SetExclusive(const bool exclusive) { (void)this->AddAttr(kExclusive, api::MakeValue(exclusive)); }

bool CumProd::GetExclusive() const {
  auto value_ptr = this->GetAttr(kExclusive);
  return GetValue<bool>(value_ptr);
}

void CumProd::SetReverse(const bool reverse) { (void)this->AddAttr(kReverse, api::MakeValue(reverse)); }

bool CumProd::GetReverse() const {
  auto value_ptr = this->GetAttr(kReverse);
  return GetValue<bool>(value_ptr);
}

void CumProd::SetAxis(const int64_t axis) { (void)this->AddAttr(kAxis, api::MakeValue(axis)); }

int64_t CumProd::GetAxis() const {
  auto value_ptr = this->GetAttr(kAxis);
  return GetValue<int64_t>(value_ptr);
}

AbstractBasePtr CumProdInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  return abstract::MakeAbstract(CumProdInferShape(primitive, input_args), CumProdInferType(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(CumProd, prim::kPrimCumProd, CumProdInfer, CumProdInferValue, true);
}  // namespace ops
}  // namespace mindspore
