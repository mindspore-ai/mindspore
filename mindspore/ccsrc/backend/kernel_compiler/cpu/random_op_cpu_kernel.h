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
#ifndef MINDSPORE_CCSRC_KERNEL_CPU_RANDOM_OP_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_CPU_RANDOM_OP_CPU_KERNEL_H_
#include <securec.h>
#include <math.h>
#include <array>
#include <iostream>

namespace mindspore {
namespace kernel {
static constexpr int gResultNum = 4;
class PhiloxGenerator {
 public:
  explicit PhiloxGenerator(uint64_t seed) {
    key_var_[0] = static_cast<uint32_t>(seed);
    key_var_[1] = static_cast<uint32_t>(seed >> 32);
    counter_[0] = 0;
    counter_[1] = 0;
    counter_[2] = static_cast<uint32_t>(seed);
    counter_[3] = static_cast<uint32_t>(seed >> 32);
  }

  void Jump() {
    if ((++counter_[0] == 0) && (++counter_[1] == 0) && (++counter_[2] == 0)) {
      ++counter_[3];
    }
  }

  void JumpStep(uint64_t step) {
    uint64_t min_counter, max_counter;
    min_counter = static_cast<uint64_t>(counter_[1]);
    min_counter = min_counter << 32;
    min_counter += counter_[0];

    max_counter = static_cast<uint64_t>(counter_[3]);
    max_counter = max_counter << 32;
    max_counter += counter_[2];
    min_counter += step;
    if (min_counter < step) {
      max_counter++;
    }
    counter_[0] = static_cast<uint32_t>(min_counter);
    counter_[1] = static_cast<uint32_t>(min_counter >> 32);
    counter_[2] = static_cast<uint32_t>(max_counter);
    counter_[3] = static_cast<uint32_t>(max_counter >> 32);
  }

  static std::array<uint32_t, 4> Compute(const std::array<uint32_t, 4> &counter_,
                                         const std::array<uint32_t, 2> &key_var_) {
    std::array<uint32_t, 4> min_value;
    std::array<uint32_t, 4> max_value;
    for (uint32_t i = 0; i < gResultNum; i += 2) {
      uint64_t temp = static_cast<uint64_t>(keyConstant[i]) * counter_[i];
      min_value[i] = static_cast<uint32_t>(temp);
      max_value[i] = static_cast<uint32_t>(temp >> 32);
    }
    std::array<uint32_t, 4> result;
    result[0] = (max_value[2] ^ counter_[1] ^ key_var_[0]);
    result[1] = min_value[2];
    result[2] = (max_value[0] ^ counter_[3] ^ key_var_[0]);
    result[3] = min_value[0];
    return result;
  }

  std::array<uint32_t, 4> operator()() {
    for (uint32_t i = 0; i < 10; i++) {
      counter_ = Compute(counter_, key_var_);
      key_var_[0] += keyConstant[1];
      key_var_[1] += keyConstant[3];
    }
    Jump();
    return counter_;
  }

 private:
  std::array<uint32_t, 4> counter_;
  std::array<uint32_t, 2> key_var_;
  static constexpr std::array<uint32_t, 4> keyConstant = {0xD2511F53, 0x9E3779B9, 0xCD9E8D57, 0xBB67AE85};
};

template <class T, typename vartype>
class NormalDistribution;
template <class T>
class NormalDistribution<T, float> {
 public:
  std::array<float, gResultNum> result;

  bool UInt32ToFloat32(uint32_t input, float *output) {
    const uint32_t temp_value = input & 0x7fffffu;
    const uint32_t exp = static_cast<uint32_t>(127);
    const uint32_t val = (exp << 23) | temp_value;
    errno_t mem_ret;
    mem_ret = memcpy_s(output, sizeof(val), &val, sizeof(val));
    if (mem_ret != EOK) {
      std::cout << "UInt32ToFloat32 memcpy is failed" << std::endl;
      return false;
    }
    *output = *output - 1.0f;
    return true;
  }

  std::array<float, gResultNum> operator()(T *generator) {
    std::array<uint32_t, 4> generate_value = (*generator)();
    const float PI = 3.14;
    for (uint32_t i = 0; i < gResultNum; i += 2) {
      float temp[2];
      UInt32ToFloat32(generate_value[i], &temp[0]);
      UInt32ToFloat32(generate_value[i + 1], &temp[1]);
      const float threshold = 1.0e-7f;
      temp[0] = temp[0] < threshold ? threshold : temp[0];
      temp[1] = temp[1] < threshold ? threshold : temp[1];
      result[i] = sqrt(-2.0 * log(temp[0])) * sin(2 * PI * temp[1]);
      result[i + 1] = sqrt(-2.0 * log(temp[0])) * cos(2 * PI * temp[1]);
    }
    return result;
  }
};

template <class T>
bool FillRandoms(PhiloxGenerator generator, float *output, int64_t vet_size, int64_t thread_Id) {
  T distribution;
  errno_t mem_ret;
  generator.JumpStep((vet_size * thread_Id + gResultNum - 1) / gResultNum);
  for (int32_t i = 0; i < vet_size; i += gResultNum) {
    auto outputResult = distribution(&generator);
    if (vet_size - i >= gResultNum) {
      mem_ret = memcpy_s(&output[i], gResultNum * sizeof(float), &outputResult[0], gResultNum * sizeof(float));
    } else {
      mem_ret = memcpy_s(&output[i], (vet_size - i) * sizeof(float), &outputResult[0], (vet_size - i) * sizeof(float));
    }
    if (mem_ret != EOK) {
      std::cout << "FillRandoms memcpy is failed" << std::endl;
      return false;
    }
  }
  return true;
}

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_CPU_RANDOM_OP_CPU_KERNEL_H_
