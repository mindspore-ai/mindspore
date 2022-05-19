/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SAMPLE_DISTORTED_BOUNDING_BOX_V2_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SAMPLE_DISTORTED_BOUNDING_BOX_V2_H_

#include <stdint.h>
#include <algorithm>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class Region {
 public:
  Region() { SetPiont(0, 0, 0, 0); }
  Region(int xmin, int ymin, int xmax, int ymax) { SetPiont(xmin, ymin, xmax, ymax); }

  void SetPiont(int xmin, int ymin, int xmax, int ymax) {
    min_x_ = xmin;
    min_y_ = ymin;
    max_x_ = xmax;
    max_y_ = ymax;
  }

  float Area() const { return static_cast<float>((max_x_ - min_x_) * (max_y_ - min_y_)); }

  Region Intersect(const Region &r) const {
    const int pmin_x = std::max(min_x_, r.min_x_);
    const int pmin_y = std::max(min_y_, r.min_y_);
    const int pmax_x = std::min(max_x_, r.max_x_);
    const int pmax_y = std::min(max_y_, r.max_y_);
    if (pmin_x > pmax_x || pmin_y > pmax_y) {
      return Region();
    } else {
      return Region(pmin_x, pmin_y, pmax_x, pmax_y);
    }
  }
  int min_x_;
  int min_y_;
  int max_x_;
  int max_y_;
};

template <typename T, size_t ElementCount>
class Array {
 public:
  Array() {
    for (size_t i = 0; i < ElementCount; ++i) {
      data_[i] = T(0);
    }
  }
  const T &operator[](size_t index) const { return data_[index]; }
  T &operator[](size_t index) { return data_[index]; }
  size_t size() const { return ElementCount; }

 private:
  T data_[ElementCount];
};

class MSPhiloxRandom {
 public:
  static constexpr size_t kIndex0 = 0;
  static constexpr size_t kIndex1 = 1;
  static constexpr size_t kIndex2 = 2;
  static constexpr size_t kIndex3 = 3;
  static constexpr size_t kKeyCount = 2;
  static constexpr size_t kResultElementCount = 4;
  static constexpr size_t loop_rounds = 10;
  /*
   * The type for the 64-bit key stored in the form of two 32-bit uint
   * that are used in the diffusion process.
   */
  using ResType = Array<uint32_t, kResultElementCount>;
  using Key = Array<uint32_t, kKeyCount>;

  MSPhiloxRandom() {}

  static constexpr int kMoveStepInBit = 32;
  explicit MSPhiloxRandom(uint64_t seed) {
    key_[kIndex0] = static_cast<uint32_t>(seed);
    key_[kIndex1] = static_cast<uint32_t>(seed >> kMoveStepInBit);
  }

  explicit MSPhiloxRandom(uint64_t seed_lo, uint64_t seed_hi) {
    key_[kIndex0] = static_cast<uint32_t>(seed_lo);
    key_[kIndex1] = static_cast<uint32_t>(seed_lo >> kMoveStepInBit);
    counter_[kIndex2] = static_cast<uint32_t>(seed_hi);
    counter_[kIndex3] = static_cast<uint32_t>(seed_hi >> kMoveStepInBit);
  }

  MSPhiloxRandom(ResType counter, Key key) : counter_(counter), key_(key) {}
  ResType const &counter() const { return counter_; }
  Key const &key() const { return key_; }

  // Skip the specified number of samples of 128-bits in the current stream.
  void Skip(uint64_t count) {
    const uint32_t count_lo = static_cast<uint32_t>(count);
    uint32_t count_hi = static_cast<uint32_t>(count >> kMoveStepInBit);

    counter_[kIndex0] += count_lo;
    if (counter_[kIndex0] < count_lo) {
      ++count_hi;
    }

    counter_[kIndex1] += count_hi;
    if (counter_[kIndex1] < count_hi) {
      if (++counter_[kIndex2] == 0) {
        ++counter_[kIndex3];
      }
    }
  }
  /*
   * Returns a group of four random numbers using the underlying Philox
   * algorithm.
   */
  ResType operator()() {
    ResType counter = counter_;
    Key key = key_;
    for (size_t i = 0; i < loop_rounds; i++) {
      counter = SingleRoundCompute(counter, key);
      RaiseKey(&key);
    }
    SkipOne();
    return counter;
  }

 private:
  // We use the same constants as recommended by the original paper.
  static constexpr uint32_t kMSPhiloxW32A = 0x9E3779B9;
  static constexpr uint32_t kMSPhiloxW32B = 0xBB67AE85;
  static constexpr uint32_t kMSPhiloxM4x32A = 0xD2511F53;
  static constexpr uint32_t kMSPhiloxM4x32B = 0xCD9E8D57;

  // Helper function to skip the next sample of 128-bits in the current stream.
  void SkipOne() {
    if (++counter_[kIndex0] == 0) {
      if (++counter_[kIndex1] == 0) {
        if (++counter_[kIndex2] == 0) {
          ++counter_[kIndex3];
        }
      }
    }
  }
  /*
   * Helper function to return the lower and higher 32-bits from two 32-bit
   * integer multiplications.
   */
  static void HighLowMultiply(uint32_t a, uint32_t b, uint32_t *result_low, uint32_t *result_high) {
    const uint64_t product = static_cast<uint64_t>(a) * static_cast<uint64_t>(b);
    *result_low = static_cast<uint32_t>(product);
    *result_high = static_cast<uint32_t>(product >> kMoveStepInBit);
  }

  // Helper function for a single round of the underlying Philox algorithm.
  static ResType SingleRoundCompute(const ResType &counter, const Key &key) {
    uint32_t low0;
    uint32_t high0;
    HighLowMultiply(kMSPhiloxM4x32A, counter[kIndex0], &low0, &high0);

    uint32_t low1;
    uint32_t high1;
    HighLowMultiply(kMSPhiloxM4x32B, counter[kIndex2], &low1, &high1);

    ResType result;
    result[kIndex0] = high1 ^ counter[kIndex1] ^ key[kIndex0];
    result[kIndex1] = low1;
    result[kIndex2] = high0 ^ counter[kIndex3] ^ key[kIndex1];
    result[kIndex3] = low0;
    return result;
  }

  void RaiseKey(Key *key) {
    (*key)[kIndex0] += kMSPhiloxW32A;
    (*key)[kIndex1] += kMSPhiloxW32B;
  }

 private:
  ResType counter_;
  Key key_;
};

class SampleDistortedBoundingBoxV2CPUKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  SampleDistortedBoundingBoxV2CPUKernelMod() = default;
  ~SampleDistortedBoundingBoxV2CPUKernelMod() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
              const std::vector<kernel::AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  int64_t seed = 0;
  int64_t seed2 = 0;
  std::vector<float> aspect_ratio_range;
  std::vector<float> area_range;
  int64_t max_attempts = 100;
  bool use_image_if_no_bounding_boxes = false;
  TypeId dtype_{kTypeUnknown};

  MSPhiloxRandom generator_;
  using ResType = Array<uint32_t, MSPhiloxRandom::kResultElementCount>;
  ResType unused_results_;
  size_t used_result_index_ = MSPhiloxRandom::kResultElementCount;

  float RandFloat();
  uint32_t Uniform(uint32_t n);
  uint64_t New64();
  void InitMSPhiloxRandom(int64_t seed, int64_t seed2);
  uint32_t GenerateSingle();
  bool SatisfiesOverlapConstraints(const Region &crop, float minimum_object_covered,
                                   const std::vector<Region> &bounding_boxes);
  bool GenerateRandomCrop(int original_width, int original_height, float min_relative_crop_area,
                          float max_relative_crop_area, float aspect_ratio, Region *crop_rect);
  template <typename T>
  void LaunchSDBBExt2(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SAMPLE_DISTORTED_BOUNDING_BOX_V2_H_
