/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/grad/avg_pool_3d_grad.h"
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
void AvgPool3DGrad::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &strides,
                         const PadMode &pad_mode, const std::vector<int64_t> &pad_list, bool ceil_mode,
                         bool count_include_pad, int64_t divisor_override, const Format &format) {
  set_kernel_size(kernel_size);
  set_strides(strides);
  set_pad_mode(pad_mode);
  set_pad_list(pad_list);
  set_ceil_mode(ceil_mode);
  set_count_include_pad(count_include_pad);
  set_divisor_override(divisor_override);
  set_format(format);
}

void AvgPool3DGrad::set_pad_list(const std::vector<int64_t> &pad_list) {
  const int64_t pad_size = 4;
  (void)CheckAndConvertUtils::CheckInteger(kPadList, SizeToLong(pad_list.size()), kEqual, pad_size, name());
  (void)AddAttr(kPadList, api::MakeValue(pad_list));
}

void AvgPool3DGrad::set_ceil_mode(bool ceil_mode) { (void)AddAttr(kCeilMode, api::MakeValue(ceil_mode)); }

void AvgPool3DGrad::set_count_include_pad(bool count_include_pad) {
  (void)AddAttr(kCountIncludePad, api::MakeValue(count_include_pad));
}

void AvgPool3DGrad::set_divisor_override(int64_t divisor_override) {
  (void)AddAttr(kDivisorOverride, api::MakeValue(divisor_override));
}

std::vector<int64_t> AvgPool3DGrad::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

bool AvgPool3DGrad::get_ceil_mode() const { return GetValue<bool>(GetAttr(kCeilMode)); }

bool AvgPool3DGrad::get_count_include_pad() const { return GetValue<bool>(GetAttr(kCountIncludePad)); }

int64_t AvgPool3DGrad::get_divisor_override() const { return GetValue<int64_t>(GetAttr(kDivisorOverride)); }

void GetTensorIntValue(const abstract::AbstractBasePtr &base, std::vector<int64_t> *value,
                       const std::string &tensor_name) {
  MS_EXCEPTION_IF_NULL(base);
  auto base_v = base->BuildValue();
  MS_EXCEPTION_IF_NULL(base_v);
  if (base->isa<abstract::AbstractTensor>()) {
    if (base_v->isa<tensor::Tensor>()) {
      *value = CheckAndConvertUtils::CheckTensorIntValue(tensor_name, base_v, kNameAvgPool3DGrad);
      (void)CheckAndConvertUtils::CheckPositiveVector(tensor_name, *value, kNameAvgPool3DGrad);
    } else {
      constexpr int64_t k5DInputDims = 5;
      value->assign(k5DInputDims, abstract::Shape::kShapeDimAny);
    }
  }
}

abstract::ShapePtr AvgPool3DGradInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInteger("input size", SizeToLong(input_args.size()), kGreaterEqual, input_num,
                                           op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  size_t grad_index = input_args.size() - 1;
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[grad_index]->GetShapeTrack())[kShape];
  constexpr int64_t k5DInputDims = 5;
  if (!IsDynamicRank(grad_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("grad_rank", SizeToLong(grad_shape.size()), kEqual, k5DInputDims, op_name);
  }
  std::vector<int64_t> origin_input_size;
  if (input_args[0]->isa<abstract::AbstractTuple>()) {  // origin_size is tuple
    origin_input_size = GetValue<std::vector<int64_t>>(input_args[0]->BuildValue());
  } else if (input_args[0]->isa<abstract::AbstractTensor>()) {
    GetTensorIntValue(input_args[0], &origin_input_size, "origin_input_shape");
  } else {
    MS_LOG(EXCEPTION) << "For '" << op_name << "', the first input data size must be a tuple or tensor, but got: "
                      << input_args[0]->BuildShape()->ToString() << ".";
  }
  return std::make_shared<abstract::Shape>(origin_input_size);
}

TypePtr AvgPool3DGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input size", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto grad_dtype = input_args.back()->BuildType();
  std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool is_gpu = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice);
  if (is_gpu) {
    valid_types = {kFloat16, kFloat32, kFloat64};
  }
  return CheckAndConvertUtils::CheckTensorTypeValid("grad", grad_dtype, valid_types, op_name);
}

MIND_API_OPERATOR_IMPL(AvgPool3DGrad, PoolGrad);
AbstractBasePtr AvgPool3DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  auto res = std::make_shared<abstract::AbstractTensor>(AvgPool3DGradInferType(primitive, input_args),
                                                        AvgPool3DGradInferShape(primitive, input_args)->shape());
  return res;
}

REGISTER_HOST_DEPENDS(kNameAvgPool3DGrad, {0});
REGISTER_PRIMITIVE_EVAL_IMPL(AvgPool3DGrad, prim::kPrimAvgPool3DGrad, AvgPool3DGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
