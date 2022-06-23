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

#include "ops/grad/max_pool_grad_grad.h"
#include <algorithm>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void MaxPoolGradGrad::set_kernel_size(const std::vector<int64_t> &kernel_size) {
  (void)this->AddAttr(kKernelSize, api::MakeValue(kernel_size));
}

std::vector<int64_t> MaxPoolGradGrad::get_kernel_size() const {
  auto value_ptr = GetAttr(kKernelSize);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void MaxPoolGradGrad::set_pad_mode(const PadMode &pad_mode) {
  int64_t swi = pad_mode;
  (void)this->AddAttr(kPadMode, api::MakeValue(swi));
}

PadMode MaxPoolGradGrad::get_pad_mode() const {
  auto value_ptr = GetAttr(kPadMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto mode_str = GetValue<std::string>(value_ptr);
  std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(), ::toupper);
  MS_EXCEPTION_IF_CHECK_FAIL((mode_str == "SAME" || mode_str == "VALID"),
                             "MaxPoolGradGrad only supports pad mode same or valid, but get " + mode_str);
  return mode_str == "SAME" ? PadMode::SAME : PadMode::VALID;
}

void MaxPoolGradGrad::set_strides(const std::vector<int64_t> &strides) {
  (void)this->AddAttr(kStrides, api::MakeValue(strides));
}

std::vector<int64_t> MaxPoolGradGrad::get_strides() const {
  auto value_ptr = GetAttr(kStrides);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

namespace {
abstract::ShapePtr MaxPoolGradGradInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  auto origin_input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  auto origin_output_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->BuildShape())[kShape];
  CheckAndConvertUtils::Check("grad_shape", origin_input_shape, kEqual, grad_shape, primitive->name(), TypeError);
  return std::make_shared<abstract::Shape>(origin_output_shape);
}

TypePtr MaxPoolGradGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  std::map<std::string, TypePtr> types;
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  (void)types.emplace("origin_input", input_args[0]->BuildType());
  (void)types.emplace("origin_output", input_args[kInputIndex1]->BuildType());
  (void)types.emplace("grad", input_args[kInputIndex2]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
}
}  // namespace

AbstractBasePtr MaxPoolGradGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = MaxPoolGradGradInferType(primitive, input_args);
  auto infer_shape = MaxPoolGradGradInferShape(primitive, input_args);
  MS_EXCEPTION_IF_NULL(infer_shape);
  return std::make_shared<abstract::AbstractTensor>(infer_type, infer_shape->shape());
}

MIND_API_OPERATOR_IMPL(MaxPoolGradGrad, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(MaxPoolGradGrad, prim::kPrimMaxPoolGradGrad, MaxPoolGradGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
