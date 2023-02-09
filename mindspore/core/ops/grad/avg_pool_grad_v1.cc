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

#include "ops/grad/avg_pool_grad_v1.h"
#include <set>
#include "ops/op_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
abstract::ShapePtr AvgPoolGradV1InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  int64_t format = CheckAndConvertUtils::GetAndCheckFormat(primitive->GetAttr("format"));
  std::vector<int64_t> kernel_size = GetValue<std::vector<int64_t>>(primitive->GetAttr(kKernelSize));

  auto pad_mode_value = (primitive->GetAttr(kPadMode));
  int64_t pad_mode;
  CheckAndConvertUtils::GetPadModEnumValue(pad_mode_value, &pad_mode, true);
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

  auto orig_shape = GetShapeValue(primitive, input_args[0]);
  return std::make_shared<abstract::Shape>(orig_shape);
}

TypePtr AvgPoolGradV1InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto name = prim->name();
  auto orig_input_shape_type = input_args[0]->BuildType();
  auto input_grad_type = input_args[1]->BuildType();
  const std::set<TypePtr> orig_input_shape_valid_type = {kInt32};
  const std::set<TypePtr> input_grad_valid_type = {kFloat16, kFloat32, kFloat64};

  (void)CheckAndConvertUtils::CheckTensorTypeValid("orig_input_shape", orig_input_shape_type,
                                                   orig_input_shape_valid_type, name);
  auto inferred_type = CheckAndConvertUtils::CheckTensorTypeValid("grad", input_grad_type, input_grad_valid_type, name);
  return inferred_type;
}

AbstractBasePtr AvgPoolGradV1Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t num_inputs = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, num_inputs, primitive->name());
  auto avgpoolgradv1_infer_type = AvgPoolGradV1InferType(primitive, input_args);
  auto avgpoolgradv1_infer_shape = AvgPoolGradV1InferShape(primitive, input_args)->shape();
  return std::make_shared<abstract::AbstractTensor>(avgpoolgradv1_infer_type, avgpoolgradv1_infer_shape);
}
MIND_API_OPERATOR_IMPL(AvgPoolGradV1, BaseOperator);

// AG means auto generated
class MIND_API AGAvgPoolGradV1Infer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    return AvgPoolGradV1InferShape(primitive, input_args);
  }

  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    return AvgPoolGradV1InferType(primitive, input_args);
  }
  AbstractBasePtr InferShapeAndType(const abstract::AnalysisEnginePtr &engine, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) const override {
    return AvgPoolGradV1Infer(engine, primitive, input_args);
  }
};

REGISTER_PRIMITIVE_OP_INFER_IMPL(AvgPoolGradV1, prim::kPrimAvgPoolGradV1, AGAvgPoolGradV1Infer, false);
}  // namespace ops
}  // namespace mindspore
