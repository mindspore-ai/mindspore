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

#include "backend/kernel_compiler/cpu/quantum/quantum_simulator/projector.h"

namespace mindspore {
namespace mindquantum {
Projector::Projector() {}
Projector::Projector(const NameType &proj_str) : proj_str_(proj_str), n_qubits_(proj_str.length()) {}
void Projector::HandleMask() {
  mask1_ = 0;
  mask2_ = 0;
  for (auto i : proj_str_) {
    if (i == '1') {
      mask1_ = mask1_ * 2 + 1;
    } else {
      mask1_ = mask1_ * 2;
    }
    if (i == '0') {
      mask2_ = mask2_ * 2;
    } else {
      mask2_ = mask2_ * 2 + 1;
    }
  }
}
Indexes Projector::GetMasks() { return {mask1_, mask2_}; }
}  // namespace mindquantum
}  // namespace mindspore
