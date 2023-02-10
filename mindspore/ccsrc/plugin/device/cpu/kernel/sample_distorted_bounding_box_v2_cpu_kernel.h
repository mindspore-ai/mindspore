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

#include <cstdint>
#include <algorithm>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/random_util.h"
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

class SampleDistortedBoundingBoxV2CPUKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  SampleDistortedBoundingBoxV2CPUKernelMod() = default;
  ~SampleDistortedBoundingBoxV2CPUKernelMod() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
              const std::vector<kernel::AddressPtr> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  int64_t seed = 0;
  int64_t seed2 = 0;
  std::vector<float> aspect_ratio_range;
  std::vector<float> area_range;
  int64_t max_attempts = 100;
  bool use_image_if_no_bounding_boxes = false;
  TypeId dtype_{kTypeUnknown};

  random::MSPhiloxRandom generator_;
  using ResType = random::Array<uint32_t, random::MSPhiloxRandom::kResultElementCount>;
  ResType unused_results_;
  size_t used_result_index_ = random::MSPhiloxRandom::kResultElementCount;

  float RandFloat();
  uint32_t Uniform(uint32_t n);
  const uint64_t New64();
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
