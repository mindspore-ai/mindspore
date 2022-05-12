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

#include "ops/grad/max_pool_grad_v1.h"
#include <set>
#include "ops/op_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
abstract::ShapePtr MaxPoolGradV1InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  int64_t format = CheckAndConvertUtils::GetAndCheckFormat(primitive->GetAttr("format"));
  auto kernel_size = GetValue<std::vector<int64_t>>(primitive->GetAttr(kKernelSize));

  auto pad_mode_value = (primitive->GetAttr(kPadMode));
  auto pad_mode = PadMode(GetValue<int64_t>(pad_mode_value));

  if (format == NHWC) {
    std::vector<int64_t> ksize_NHWC = {kernel_size[0], kernel_size[1], kernel_size[2], kernel_size[3]};
    (void)primitive->AddAttr("ksize", MakeValue(ksize_NHWC));
    (void)primitive->DelAttr("data_format");
    (void)primitive->AddAttr("data_format", MakeValue("NHWC"));
  } else if (format == NCHW) {
    std::vector<int64_t> ksize_NCHW = {kernel_size[0], kernel_size[1], kernel_size[2], kernel_size[3]};
    (void)primitive->AddAttr("ksize", MakeValue(ksize_NCHW));
    (void)primitive->DelAttr("data_format");
    (void)primitive->AddAttr("data_format", MakeValue("NCHW"));
  }
  if (pad_mode == VALID) {
    (void)primitive->AddAttr("padding", MakeValue("VALID"));
  } else if (pad_mode == SAME) {
    (void)primitive->AddAttr("padding", MakeValue("SAME"));
  }

  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  return std::make_shared<abstract::Shape>(in_shape);
}
TypePtr MaxPoolGradV1InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto name = prim->name();
  const std::set<TypePtr> valid_types = {kInt8,    kInt16, kInt32,  kInt64,  kFloat16, kFloat32,
                                         kFloat64, kUInt8, kUInt16, kUInt32, kUInt64};
  auto orig_input_type = input_args[0]->BuildType();
  auto orig_output_type = input_args[0]->BuildType();
  auto grad_type = input_args[0]->BuildType();
  auto inferred_type = CheckAndConvertUtils::CheckTensorTypeValid("orig_input", orig_input_type, valid_types, name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("orig_output", orig_output_type, valid_types, name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("grad", grad_type, valid_types, name);
  return inferred_type;
}
AbstractBasePtr MaxPoolGradV1Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto maxpoolgradv1_infer_type = MaxPoolGradV1InferType(primitive, input_args);
  auto maxpoolgradv1_infer_shape = MaxPoolGradV1InferShape(primitive, input_args)->shape();
  return std::make_shared<abstract::AbstractTensor>(maxpoolgradv1_infer_type, maxpoolgradv1_infer_shape);
}
MIND_API_OPERATOR_IMPL(MaxPoolGradV1, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(MaxPoolGradV1, prim::kPrimMaxPoolGradV1, MaxPoolGradV1Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
