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

#include "ops/softmax_cross_entropy_with_logits.h"
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
AbstractBasePtr SoftmaxCrossEntropyWithLogitsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto softmax_prim = primitive->cast<PrimSoftmaxCrossEntropyWithLogitsPtr>();
  MS_EXCEPTION_IF_NULL(softmax_prim);
  auto prim_name = softmax_prim->name();
  CheckAndConvertUtils::CheckInteger("softmax_cross_entropy_with_logics_infer", input_args.size(), kEqual, 2,
                                     prim_name);

  // Infer shape
  auto logits_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("logits_shape", input_args[0]->BuildShape(), prim_name);
  auto labels_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("labels_shape", input_args[1]->BuildShape(), prim_name);
  CheckAndConvertUtils::Check("logits shape", logits_shape, kEqual, "labels shape", labels_shape, prim_name, TypeError);
  std::vector<int64_t> loss_shape = {logits_shape[0]};
  auto dlogits_shape = logits_shape;

  // Infer type
  const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32};
  std::map<std::string, TypePtr> args;
  args.emplace("logits_type", input_args[0]->BuildType());
  args.emplace("labels_type", input_args[1]->BuildType());
  CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  auto logits_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();

  auto output0 = std::make_shared<abstract::AbstractTensor>(logits_type, loss_shape);
  auto output1 = std::make_shared<abstract::AbstractTensor>(logits_type, dlogits_shape);
  AbstractBasePtrList output = {output0, output1};
  return std::make_shared<abstract::AbstractTuple>(output);
}
REGISTER_PRIMITIVE_C(kNameSoftmaxCrossEntropyWithLogits, SoftmaxCrossEntropyWithLogits);
}  // namespace ops
}  // namespace mindspore
