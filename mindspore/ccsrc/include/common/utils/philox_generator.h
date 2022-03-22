/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PHILOX_GENERATOR_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PHILOX_GENERATOR_H_

#include <securec.h>
#include <math.h>
#include <array>
#include "utils/log_adapter.h"
#include "utils/convert_utils_base.h"
#include "include/common/visible.h"

namespace mindspore {
static constexpr int kResultNum = 4;
class COMMON_EXPORT PhiloxGenerator {
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

  std::array<uint32_t, kResultNum> Compute(const std::array<uint32_t, kResultNum> &counter,
                                           const std::array<uint32_t, 2> &key_var) const;

  std::array<uint32_t, kResultNum> operator()();

 private:
  std::array<uint32_t, kResultNum> counter_;
  std::array<uint32_t, 2> key_var_;
  static constexpr std::array<uint32_t, kResultNum> keyConstant = {0xD2511F53, 0x9E3779B9, 0xCD9E8D57, 0xBB67AE85};
};

template <class T>
bool FillRandoms(PhiloxGenerator generator, float *output, int64_t vet_size, int64_t thread_Id) {
  T distribution;
  errno_t mem_ret;
  generator.JumpStep(LongToSize((vet_size * thread_Id + kResultNum - 1) / kResultNum));
  for (int32_t i = 0; i < vet_size; i += kResultNum) {
    auto outputResult = distribution(&generator);
    size_t max_length = 0;
    if (vet_size - i >= kResultNum) {
      max_length = kResultNum * sizeof(float);
      mem_ret = memcpy_s(&output[i], max_length, &outputResult[0], max_length);
    } else {
      max_length = LongToSize((vet_size - i) * sizeof(float));
      mem_ret = memcpy_s(&output[i], max_length, &outputResult[0], max_length);
    }
    if (mem_ret != EOK) {
      MS_LOG(ERROR) << "FillRandoms memcpy is failed";
      return false;
    }
  }
  return true;
}
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_PHILOX_GENERATOR_H_
