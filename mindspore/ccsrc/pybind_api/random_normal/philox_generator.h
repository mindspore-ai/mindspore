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
#ifndef PYBIND_API_API_IR_RANDOM_NORMAL_PHILOX_GENERATOR_H_
#define PYBIND_API_API_IR_RANDOM_NORMAL_PHILOX_GENERATOR_H_
#include <securec.h>
#include <math.h>
#include <array>

namespace mindspore {
static constexpr int gResultNum = 4;
class PhiloxGenerator {
 public:
  explicit PhiloxGenerator(uint64_t seed_) {
    key_var_[0] = static_cast<uint32_t>(seed_);
    key_var_[1] = static_cast<uint32_t>(seed_ >> 32);
    counter_[0] = 0;
    counter_[1] = 0;
    counter_[2] = static_cast<uint32_t>(seed_);
    counter_[3] = static_cast<uint32_t>(seed_ >> 32);
  }

  explicit PhiloxGenerator(uint64_t seed_, uint64_t seed2_) {
    key_var_[0] = static_cast<uint32_t>(seed_);
    key_var_[1] = static_cast<uint32_t>(seed_ >> 32);
    counter_[0] = 0;
    counter_[1] = 0;
    counter_[2] = static_cast<uint32_t>(seed2_);
    counter_[3] = static_cast<uint32_t>(seed2_ >> 32);
  }

  ~PhiloxGenerator() = default;

  void Jump();

  void JumpStep(uint64_t step);

  std::array<uint32_t, gResultNum> Compute(const std::array<uint32_t, gResultNum> &counter,
                                           const std::array<uint32_t, 2> &key_var) const;

  std::array<uint32_t, gResultNum> operator()();

 private:
  std::array<uint32_t, gResultNum> counter_;
  std::array<uint32_t, 2> key_var_;
  static constexpr std::array<uint32_t, gResultNum> keyConstant = {0xD2511F53, 0x9E3779B9, 0xCD9E8D57, 0xBB67AE85};
};
}  // namespace mindspore

#endif  // PYBIND_API_API_IR_RANDOM_NORMAL_PHILOX_GENERATOR_H_
