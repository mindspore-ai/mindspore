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

#include "ops/grad/sigmoid_cross_entropy_with_logits_grad.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
AbstractBasePtr SigmoidCrossEntropyWithLogitsGradInfer(const abstract::AnalysisEnginePtr &,
                                                       const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("sigmoid_cross_entropy_with_logits_grad_infer",
                                           SizeToLong(input_args.size()), kEqual, input_num, prim_name);

  // Infer Shape
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto y_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto dout_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  CheckAndConvertUtils::Check("x_shape", x_shape, kEqual, "y_shape", y_shape, prim_name, TypeError);
  CheckAndConvertUtils::Check("x_shape", x_shape, kEqual, "dout_shape", dout_shape, prim_name, TypeError);

  // Infer type
  const std::set<TypePtr> valid_types = {kBool,   kInt,    kInt8,   kInt16, kInt32,   kInt64,   kUInt,    kUInt8,
                                         kUInt16, kUInt32, kUInt64, kFloat, kFloat16, kFloat32, kFloat64, kComplex64};
  std::map<std::string, TypePtr> args;
  (void)args.emplace("x_type", input_args[kInputIndex0]->BuildType());
  (void)args.emplace("y_type", input_args[kInputIndex1]->BuildType());
  (void)args.emplace("dout_type", input_args[kInputIndex2]->BuildType());
  auto dout_type = CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  return std::make_shared<abstract::AbstractTensor>(dout_type, x_shape);
}
REGISTER_PRIMITIVE_C(kNameSigmoidCrossEntropyWithLogitsGrad, SigmoidCrossEntropyWithLogitsGrad);
}  // namespace ops
}  // namespace mindspore
