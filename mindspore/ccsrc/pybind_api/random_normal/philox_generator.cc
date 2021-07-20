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
#include "pybind_api/random_normal/philox_generator.h"

namespace mindspore {
static constexpr uint64_t kShiftNum = 32;
static constexpr uint64_t kGenerateNum = 10;
void PhiloxGenerator::Jump() {
  if ((++counter_[0] == 0) && (++counter_[1] == 0) && (++counter_[2] == 0)) {
    ++counter_[3];
  }
}

void PhiloxGenerator::JumpStep(uint64_t step) {
  uint64_t min_counter, max_counter;
  min_counter = static_cast<uint64_t>(counter_[1]);
  min_counter = min_counter << kShiftNum;
  min_counter += counter_[0];

  max_counter = static_cast<uint64_t>(counter_[3]);
  max_counter = max_counter << kShiftNum;
  max_counter += counter_[2];
  min_counter += step;
  if (min_counter < step) {
    max_counter++;
  }
  counter_[0] = static_cast<uint32_t>(min_counter);
  counter_[1] = static_cast<uint32_t>(min_counter >> kShiftNum);
  counter_[2] = static_cast<uint32_t>(max_counter);
  counter_[3] = static_cast<uint32_t>(max_counter >> kShiftNum);
}

std::array<uint32_t, gResultNum> PhiloxGenerator::Compute(const std::array<uint32_t, gResultNum> &counter,
                                                          const std::array<uint32_t, 2> &key_var) const {
  std::array<uint32_t, gResultNum> min_value;
  std::array<uint32_t, gResultNum> max_value;
  for (size_t i = 0; i < gResultNum; i += 2) {
    uint64_t temp = static_cast<uint64_t>(keyConstant[i]) * counter[i];
    min_value[i] = static_cast<uint32_t>(temp);
    max_value[i] = static_cast<uint32_t>(temp >> kShiftNum);
  }
  std::array<uint32_t, gResultNum> result;
  result[0] = (max_value[2] ^ counter[1] ^ key_var[0]);
  result[1] = min_value[2];
  result[2] = (max_value[0] ^ counter[3] ^ key_var[0]);
  result[3] = min_value[0];
  return result;
}

std::array<uint32_t, gResultNum> PhiloxGenerator::operator()() {
  for (size_t i = 0; i < kGenerateNum; i++) {
    counter_ = Compute(counter_, key_var_);
    key_var_[0] += keyConstant[1];
    key_var_[1] += keyConstant[3];
  }
  Jump();
  return counter_;
}
}  // namespace mindspore
