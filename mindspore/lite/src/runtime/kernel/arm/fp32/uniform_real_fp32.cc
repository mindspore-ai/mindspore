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
#include "src/runtime/kernel/arm/fp32/uniform_real_fp32.h"
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "src/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_UniformReal;

namespace mindspore::kernel {

class PhiloxRandom {
 public:
  explicit PhiloxRandom(uint64_t seed_lo, uint64_t seed_hi) {
    key_[0] = static_cast<uint32_t>(seed_lo);
    key_[1] = static_cast<uint32_t>(seed_lo >> 32);
    counter_[2] = static_cast<uint32_t>(seed_hi);
    counter_[3] = static_cast<uint32_t>(seed_hi >> 32);
  }
  ~PhiloxRandom() = default;

  // Skip the specified number of samples of 128-bits in the current stream.
  void Skip(uint64_t count) {
    const uint32_t count_lo = static_cast<uint32_t>(count);
    uint32_t count_hi = static_cast<uint32_t>(count >> 32);

    counter_[0] += count_lo;
    if (counter_[0] < count_lo) {
      ++count_hi;
    }

    counter_[1] += count_hi;
    if (counter_[1] < count_hi) {
      if (++counter_[2] == 0) {
        ++counter_[3];
      }
    }
  }

  // Returns a group of four random numbers using the underlying Philox
  // algorithm.
  std::vector<uint32_t> operator()() {
    std::vector<uint32_t> counter = counter_;
    std::vector<uint32_t> key = key_;

    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);
    RaiseKey(&key);
    counter = ComputeSingleRound(counter, key);

    SkipOne();

    return counter;
  }

 private:
  // We use the same constants as recommended by the original paper.
  static constexpr uint32_t kPhiloxW32A = 0x9E3779B9;
  static constexpr uint32_t kPhiloxW32B = 0xBB67AE85;
  static constexpr uint32_t kPhiloxM4x32A = 0xD2511F53;
  static constexpr uint32_t kPhiloxM4x32B = 0xCD9E8D57;

  // Helper function to skip the next sample of 128-bits in the current stream.
  void SkipOne() {
    if (++counter_[0] == 0) {
      if (++counter_[1] == 0) {
        if (++counter_[2] == 0) {
          ++counter_[3];
        }
      }
    }
  }

  static void MultiplyHighLow(uint32_t a, uint32_t b, uint32_t *result_low, uint32_t *result_high) {
    const uint64_t product = static_cast<uint64_t>(a) * b;
    *result_low = static_cast<uint32_t>(product);
    *result_high = static_cast<uint32_t>(product >> 32);
  }

  // Helper function for a single round of the underlying Philox algorithm.
  static std::vector<uint32_t> ComputeSingleRound(const std::vector<uint32_t> &counter,
                                                  const std::vector<uint32_t> &key) {
    uint32_t lo0;
    uint32_t hi0;
    MultiplyHighLow(kPhiloxM4x32A, counter[0], &lo0, &hi0);

    uint32_t lo1;
    uint32_t hi1;
    MultiplyHighLow(kPhiloxM4x32B, counter[2], &lo1, &hi1);

    std::vector<uint32_t> result = {0, 0, 0, 0};
    result[0] = hi1 ^ counter[1] ^ key[0];
    result[1] = lo1;
    result[2] = hi0 ^ counter[3] ^ key[1];
    result[3] = lo0;
    return result;
  }

  void RaiseKey(std::vector<uint32_t> *key) {
    (*key)[0] += kPhiloxW32A;
    (*key)[1] += kPhiloxW32B;
  }

 private:
  std::vector<uint32_t> counter_ = {0, 0, 0, 0};
  std::vector<uint32_t> key_ = {0, 0};
};

float uint32ToFloat(uint32_t x) {
  const uint32_t man = x & 0x7fffffu;  // 23 bit mantissa
  const uint32_t exp = static_cast<uint32_t>(127);
  const uint32_t val = (exp << 23) | man;

  // Assumes that endian-ness is same for float and uint32_t.
  float result;
  memcpy(&result, &val, sizeof(val));
  return result - 1.0f;
}

void GetPhiloxRandomFloat(float *data, size_t length, int seed, int seed2) {
  PhiloxRandom philoxRandom(seed, seed2);
  if (length < 4) {
    auto randNum = philoxRandom.operator()();
    for (size_t i = 0; i < length; i++) {
      data[i] = uint32ToFloat(randNum[i]);
    }
  } else {
    auto randNum = philoxRandom.operator()();
    data[0] = uint32ToFloat(randNum[0]);
    data[1] = uint32ToFloat(randNum[1]);
    data[2] = uint32ToFloat(randNum[2]);
    data[3] = uint32ToFloat(randNum[3]);
    for (size_t i = 1; i < length / 4; i++) {
      philoxRandom.Skip(0);
      randNum = philoxRandom.operator()();
      data[4 * i] = uint32ToFloat(randNum[0]);
      data[4 * i + 1] = uint32ToFloat(randNum[1]);
      data[4 * i + 2] = uint32ToFloat(randNum[2]);
      data[4 * i + 3] = uint32ToFloat(randNum[3]);
    }
    philoxRandom.Skip(0);
    randNum = philoxRandom.operator()();
    for (size_t i = 0; i < length % 4; i++) {
      data[4 * (length / 4) + i] = uint32ToFloat(randNum[i]);
    }
  }
}

int UniformRealCPUKernel::Init() { return RET_OK; }

int UniformRealCPUKernel::ReSize() { return RET_OK; }

int UniformRealCPUKernel::Run() {
  auto output0 = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  MS_ASSERT(output0);
  if (seed_ < 0 || seed2_ < 0) {
    MS_LOG(ERROR) << "seed_:" << seed_ << " and seed2_:" << seed2_ << " must be greater than 0!";
    return RET_ERROR;
  }
  if (seed_ > 0 && seed2_ > 0) {
    GetPhiloxRandomFloat(output0, out_tensors_.at(0)->ElementsNum(), seed_, seed2_);
    return RET_OK;
  }
  std::srand(seed_ || seed2_);
  for (int i = 0; i < out_tensors_.at(0)->ElementsNum(); ++i) {
    output0[i] = static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_UniformReal, LiteKernelCreator<UniformRealCPUKernel>)
}  // namespace mindspore::kernel
