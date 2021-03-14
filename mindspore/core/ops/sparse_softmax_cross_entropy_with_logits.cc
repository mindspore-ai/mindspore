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
  this->AddAttr(kIsGrad, MakeValue(is_grad));
}

bool SparseSoftmaxCrossEntropyWithLogits::get_is_grad() const {
  auto value_ptr = GetAttr(kIsGrad);
  return GetValue<bool>(value_ptr);
}

AbstractBasePtr SparseSoftmaxCrossEntropyWithLogitsInfer(const abstract::AnalysisEnginePtr &,
                                                         const PrimitivePtr &primitive,
                                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto sparse_softmax_cross_entropy_prim = primitive->cast<PrimSparseSoftmaxCrossEntropyWithLogitsPtr>();
  MS_EXCEPTION_IF_NULL(sparse_softmax_cross_entropy_prim);
  auto prim_name = sparse_softmax_cross_entropy_prim->name();
  CheckAndConvertUtils::CheckInteger("input numbers", input_args.size(), kEqual, 2, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  // infer shape
  auto input_shape =
    CheckAndConvertUtils::ConvertShapePtrToShape("input_shape", input_args[0]->BuildShape(), prim_name);
  std::vector<int64_t> output_shape;
  if (sparse_softmax_cross_entropy_prim->get_is_grad() != 0) {
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
