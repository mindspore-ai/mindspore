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

#include "ops/sparse_softmax_cross_entropy.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void SparseSoftmaxCrossEntropy::Init(const bool grad) { this->set_grad(grad); }

void SparseSoftmaxCrossEntropy::set_grad(const bool grad) { this->AddAttr(kGrad, MakeValue(grad)); }

bool SparseSoftmaxCrossEntropy::get_grad() const {
  auto value_ptr = GetAttr(kGrad);
  return GetValue<bool>(value_ptr);
}

AbstractBasePtr SparseSoftmaxCrossEntropyInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto sparse_softmax_cross_entropy_prim = primitive->cast<PrimSparseSoftmaxCrossEntropyPtr>();
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
  if (sparse_softmax_cross_entropy_prim->get_grad() != 0) {
    output_shape = input_shape;
  } else {
    output_shape.push_back(1);
  }
  // infer type
  auto output_type = input_args[0]->BuildType()->cast<TensorTypePtr>()->element();
  return std::make_shared<abstract::AbstractTensor>(output_type, output_shape);
}
REGISTER_PRIMITIVE_EVAL_IMPL(SparseSoftmaxCrossEntropy, prim::kPrimSparseSoftmaxCrossEntropy,
                             SparseSoftmaxCrossEntropyInfer);
REGISTER_PRIMITIVE_C(kNameSparseSoftmaxCrossEntropy, SparseSoftmaxCrossEntropy);
}  // namespace ops
}  // namespace mindspore
