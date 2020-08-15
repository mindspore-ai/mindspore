/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "c_ops/fake_quant_with_min_max_vars.h"

namespace mindspore {
#ifdef PRIMITIVE_WRITEABLE
bool FakeQuantWithMinMaxVars::GetNarrowRange() const {
  return this->primitive->value.AsFakeQuantWithMinMaxVars()->narrowRange;
}
int FakeQuantWithMinMaxVars::GetNumBits() const { return this->primitive->value.AsFakeQuantWithMinMaxVars()->numBits; }

void FakeQuantWithMinMaxVars::SetNarrowRange(bool narrow_range) {
  this->primitive->value.AsFakeQuantWithMinMaxVars()->narrowRange = narrow_range;
}
void FakeQuantWithMinMaxVars::SetNumBits(int num_bits) {
  this->primitive->value.AsFakeQuantWithMinMaxVars()->numBits = num_bits;
}

#else

bool FakeQuantWithMinMaxVars::GetNarrowRange() const {
  return this->primitive->value_as_FakeQuantWithMinMaxVars()->narrowRange();
}
int FakeQuantWithMinMaxVars::GetNumBits() const {
  return this->primitive->value_as_FakeQuantWithMinMaxVars()->numBits();
}

void FakeQuantWithMinMaxVars::SetNarrowRange(bool narrow_range) {}
void FakeQuantWithMinMaxVars::SetNumBits(int num_bits) {}
#endif
}  // namespace mindspore
