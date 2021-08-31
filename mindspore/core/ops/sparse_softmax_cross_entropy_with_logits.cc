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
#include <map>
#include <string>
#include <vector>

#include "ops/sparse_softmax_cross_entropy_with_logits.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void SparseSoftmaxCrossEntropyWithLogits::Init(const bool is_grad) { this->set_is_grad(is_grad); }

void SparseSoftmaxCrossEntropyWithLogits::set_is_grad(const bool is_grad) {
  (void)this->AddAttr(kIsGrad, MakeValue(is_grad));
}

bool SparseSoftmaxCrossEntropyWithLogits::get_is_grad() const { return GetValue<bool>(GetAttr(kIsGrad)); }

AbstractBasePtr SparseSoftmaxCrossEntropyWithLogitsInfer(const abstract::AnalysisEnginePtr &,
                                                         const PrimitivePtr &primitive,
                                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, input_num,
                                           prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // infer shape
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  std::vector<int64_t> output_shape;
  if (GetValue<bool>(primitive->GetAttr(kIsGrad)) != 0) {
    output_shape = input_shape;
  } else {
    output_shape.push_back(1);
  }
  // infer type
  auto output_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  return std::make_shared<abstract::AbstractTensor>(output_type, output_shape);
}
REGISTER_PRIMITIVE_C(kNameSparseSoftmaxCrossEntropyWithLogits, SparseSoftmaxCrossEntropyWithLogits);
}  // namespace ops
}  // namespace mindspore
