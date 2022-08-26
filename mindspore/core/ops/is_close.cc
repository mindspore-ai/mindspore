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

#include "ops/is_close.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr IsCloseInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool is_ascend = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  if (is_ascend) {
    const int MAX = 0x3fffffff;
    auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
    auto other_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
    auto input_rank = SizeToLong(input_shape.size());
    auto other_rank = SizeToLong(other_shape.size());
    CheckAndConvertUtils::Check("input rank", input_rank, kEqual, other_rank, op_name);
    int64_t input_size = 1, other_size = 1;
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_size *= input_shape[i];
      other_size *= other_shape[i];
      if (input_shape[i] != other_shape[i] && (input_shape[i] != 1 || other_shape[i] != 1)) {
        MS_EXCEPTION(ValueError) << "For '" << op_name
                                 << "', the size of tensor 'input' must match the size of tensor 'other' at the " << i
                                 << "th dimension, but got 'input' size: " << input_shape[i]
                                 << ", 'other' size: " << other_shape[i] << ".";
      }
    }
    if (input_size > MAX) {
      MS_EXCEPTION(ValueError) << "For '" << op_name
                               << "', the size of tensor 'input' must be less than [2147483648], but got: "
                               << input_size << ".";
    }
    if (other_size > MAX) {
      MS_EXCEPTION(ValueError) << "For '" << op_name
                               << "', the size of tensor 'other' must be less than [2147483648], but got: "
                               << other_size << ".";
    }
  }
  return BroadCastInferShape(op_name, input_args);
}

TypePtr IsCloseInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kInt8, kInt16, kInt32, kInt64, kUInt8};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input", input_args[0]->BuildType());
  (void)types.emplace("other", input_args[1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input_args[0]->BuildType(), valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("other", input_args[1]->BuildType(), valid_types, op_name);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return std::make_shared<TensorType>(kBool);
}
}  // namespace

void IsClose::Init(const float rtol, const float atol, const bool equal_nan) {
  this->set_rtol(rtol);
  this->set_atol(atol);
  this->set_equal_nan(equal_nan);
}

void IsClose::set_rtol(const float rtol) { (void)this->AddAttr(kRtol, api::MakeValue(rtol)); }

void IsClose::set_atol(const float atol) { (void)this->AddAttr(kAtol, api::MakeValue(atol)); }

void IsClose::set_equal_nan(const bool equal_nan) { (void)this->AddAttr(kEqualNan, api::MakeValue(equal_nan)); }

float IsClose::get_rtol() const {
  auto value_ptr = this->GetAttr(kRtol);
  return GetValue<float>(value_ptr);
}

float IsClose::get_atol() const {
  auto value_ptr = this->GetAttr(kAtol);
  return GetValue<float>(value_ptr);
}

bool IsClose::get_equal_nan() const {
  auto value_ptr = this->GetAttr(kEqualNan);
  return GetValue<bool>(value_ptr);
}
MIND_API_OPERATOR_IMPL(IsClose, BaseOperator);

AbstractBasePtr IsCloseInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infertype = IsCloseInferType(primitive, input_args);
  auto infershape = IsCloseInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}
REGISTER_PRIMITIVE_EVAL_IMPL(IsClose, prim::kPrimIsClose, IsCloseInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
