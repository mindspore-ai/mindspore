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

REGISTER_PRIMITIVE_C(kNameSparseSoftmaxCrossEntropyWithLogits, SparseSoftmaxCrossEntropyWithLogits);
}  // namespace ops
}  // namespace mindspore
