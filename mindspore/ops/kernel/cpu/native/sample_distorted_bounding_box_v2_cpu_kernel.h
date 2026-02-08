/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include <map>
#include <vector>
#include "kernel/cpu/cpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/philox_random.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace sample_distorted_bounding_box_v2_cpu {
class Region {
 public:
  Region() {}
  Region(int x_min, int y_min, int x_max, int y_max) : x_min_(x_min), y_min_(y_min), x_max_(x_max), y_max_(y_max) {}

  float Area() const { return static_cast<float>((x_max_ - x_min_) * (y_max_ - y_min_)); }

  Region Intersect(const Region &r) const {
    const int tmp_min_x = std::max(x_min_, r.x_min_);
    const int tmp_min_y = std::max(y_min_, r.y_min_);
    const int tmp_max_x = std::min(x_max_, r.x_max_);
    const int tmp_max_y = std::min(y_max_, r.y_max_);
    if (tmp_min_x > tmp_max_x || tmp_min_y > tmp_max_y) {
      return Region();
    } else {
      return Region(tmp_min_x, tmp_min_y, tmp_max_x, tmp_max_y);
    }
  }
  int x_min_ = 0;
  int y_min_ = 0;
  int x_max_ = 0;
  int y_max_ = 0;
};

class SampleDistortedBoundingBoxV2CPUKernelMod : public NativeCpuKernelMod {
 public:
  SampleDistortedBoundingBoxV2CPUKernelMod() = default;
  ~SampleDistortedBoundingBoxV2CPUKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<kernel::KernelTensor *> &inputs, const std::vector<kernel::KernelTensor *> &workspace,
              const std::vector<kernel::KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  int64_t seed_{0};
  int64_t seed2_{0};
  std::vector<float> aspect_ratio_range_;
  std::vector<float> area_range_;
  int64_t max_attempts_{100};
  bool use_image_if_no_bounding_boxes_{false};
  TypeId dtype_{kTypeUnknown};

  random::PhiloxRandom generator_;
  using ResType = random::Array<uint32_t, random::PhiloxRandom::kResultElementCount>;
  ResType unused_results_;
  size_t used_result_index_ = random::PhiloxRandom::kResultElementCount;

  float RandFloat();
  uint32_t Uniform(uint32_t n);
  const uint64_t New64();
  void InitMSPhiloxRandom(int64_t seed, int64_t seed2);
  uint32_t GenerateSingle();
  bool SatisfiesOverlapConstraints(const Region &crop, float minimum_object_covered,
                                   const std::vector<Region> &bounding_boxes) const;
  void PickRandomOffsets(int width, int height, int original_width, int original_height, int *x, int *y);
  bool GenerateRandomCrop(int original_width, int original_height, float min_relative_crop_area,
                          float max_relative_crop_area, float aspect_ratio, Region *crop_rect);
  template <typename T>
  void LaunchSDBBExt2(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
};
}  // namespace sample_distorted_bounding_box_v2_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SAMPLE_DISTORTED_BOUNDING_BOX_V2_H_
