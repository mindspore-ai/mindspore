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

#include "c_ops/binary_cross_entropy.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "c_ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {

void BinaryCrossEntropy::set_reduction(const std::string &reduction) {
  CheckAndConvertUtils::CheckString(kReduction, reduction, {"none", "mean", "sum"}, this->name());
  this->AddAttr(kReduction, MakeValue(reduction));
}
std::string BinaryCrossEntropy::get_reduction() const {
  auto value_ptr = GetAttr(kReduction);
  return GetValue<std::string>(value_ptr);
}

void BinaryCrossEntropy::Init(const std::string &reduction) { this->set_reduction(reduction); }
REGISTER_PRIMITIVE_C(kNameBinaryCrossEntropy, BinaryCrossEntropy);
}  // namespace mindspore
