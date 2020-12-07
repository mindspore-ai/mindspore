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

#include "c_ops/binary_cross_entropy_grad.h"

namespace mindspore {
void BinaryCrossEntropyGrad::Init(const std::string &reduction) { set_reduction(reduction); }

void BinaryCrossEntropyGrad::set_reduction(const std::string &reduction) {
  CheckAndConvertUtils::CheckString(kReduction, reduction, {"none", "mean", "sum"}, name());
  this->AddAttr(kReduction, MakeValue(reduction));
}
std::string BinaryCrossEntropyGrad::get_reduction() const {
  auto value_ptr = GetAttr(kReduction);
  return GetValue<std::string>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameBinaryCrossEntropyGrad, BinaryCrossEntropyGrad);
}  // namespace mindspore
