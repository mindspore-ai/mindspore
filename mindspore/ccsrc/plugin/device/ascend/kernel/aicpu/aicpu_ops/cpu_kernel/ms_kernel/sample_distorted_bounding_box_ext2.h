/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_SAMPLE_DISTORTED_BOUNDING_BOX_EXT2_H_
#define AICPU_KERNELS_NORMALIZED_SAMPLE_DISTORTED_BOUNDING_BOX_EXT2_H_

#include <algorithm>
#include <vector>
#include "inc/cpu_ops_kernel.h"
#include "utils/philox_random.h"

class Rectangle {
 public:
  Rectangle() { Set(0, 0, 0, 0); }
  Rectangle(int xmin, int ymin, int xmax, int ymax) { Set(xmin, ymin, xmax, ymax); }

  void Set(int xmin, int ymin, int xmax, int ymax) {
    min_x_ = xmin;
    min_y_ = ymin;
    max_x_ = xmax;
    max_y_ = ymax;
  }

  bool IsEmpty() const { return min_x_ > max_x_ || min_y_ > max_y_; }
  float Area() const { return static_cast<float>((max_x_ - min_x_) * (max_y_ - min_y_)); }

  Rectangle Intersect(const Rectangle &r) const {
    const int pmin_x = std::max(min_x_, r.min_x_);
    const int pmin_y = std::max(min_y_, r.min_y_);
    const int pmax_x = std::min(max_x_, r.max_x_);
    const int pmax_y = std::min(max_y_, r.max_y_);
    if (pmin_x > pmax_x || pmin_y > pmax_y) {
      return Rectangle();
    } else {
      return Rectangle(pmin_x, pmin_y, pmax_x, pmax_y);
    }
  }

  int min_x_;
  int min_y_;
  int max_x_;
  int max_y_;
};

namespace aicpu {
class SDBBExt2CpuKernel : public CpuKernel {
 public:
  SDBBExt2CpuKernel() = default;
  ~SDBBExt2CpuKernel() override = default;

  static const int kResultTypeNum = 4;
  static const int kKeyNum = 2;
  using ResultType = random::Array<uint32_t, kResultTypeNum>;
  using ResultElementType = uint32_t;
  using Key = random::Array<uint32_t, kKeyNum>;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  int seed;
  int seed2;
  std::vector<float> aspect_ratio_range;
  std::vector<float> area_range;
  int max_attempts;
  bool use_image_if_no_bounding_boxes;

  random::PhiloxRandom generator_;

  float RandFloat();
  uint32_t Uniform(uint32_t n);

  uint64_t New64();
  void InitPhiloxRandom(int64_t seed, int64_t seed2);
  random::PhiloxRandom::ResultType unused_results_;
  int used_result_index_ = random::PhiloxRandom::kResultElementCount;
  ResultElementType GenerateSingle();

  // Image
  bool SatisfiesOverlapConstraints(const Rectangle &crop, float minimum_object_covered,
                                   const std::vector<Rectangle> &bounding_boxes);
  bool GenerateRandomCrop(int original_width, int original_height, float min_relative_crop_area,
                          float max_relative_crop_area, float aspect_ratio, Rectangle *crop_rect);

  uint32_t SDBBExt2Check(const CpuKernelContext &ctx);

  template <typename T>
  uint32_t SDBBExt2Compute(CpuKernelContext &ctx);
};
}  // namespace aicpu
#endif
