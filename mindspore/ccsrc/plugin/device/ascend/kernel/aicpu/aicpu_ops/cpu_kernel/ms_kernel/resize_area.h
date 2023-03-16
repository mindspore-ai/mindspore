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
#ifndef AICPU_KERNELS_NORMALIZED_RESIZE_AREA_H_
#define AICPU_KERNELS_NORMALIZED_RESIZE_AREA_H_

#include <string>
#include "Eigen/Core"
#include "cpu_ops_kernel.h"

namespace aicpu {
// weight data of every pixel
struct ResizeAreaCachedInterpolation {
  size_t start;
  size_t end;
  float start_scale;
  float end_minus_one_scale;
  bool needs_bounding = true;
};

struct ResizeAreaSt {
  void CalSt(CpuKernelContext &ctx, std::vector<int64_t> &in_shape1, bool align_corners);
  size_t batch_size;
  size_t channels;
  size_t in_height;
  size_t in_width;
  size_t out_height;
  size_t out_width;
  float height_scale;
  float width_scale;
};

class ResizeAreaCpuKernel : public CpuKernel {
 public:
  ~ResizeAreaCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t DoCompute(const ResizeAreaSt &st, std::vector<ResizeAreaCachedInterpolation> &x_interps,
                     int64_t kKnownNumChannels, CpuKernelContext &ctx);
  template <bool NeedsXBounding, typename T>
  void ComputePatchSumOf3Channels(float scale, const ResizeAreaSt &st, const std::vector<const T *> &y_ptrs,
                                  const std::vector<float> &y_scales, const ResizeAreaCachedInterpolation &x_interp,
                                  float *&output_patch_ptr);
  template <bool NeedsXBounding, typename T>
  void ComputePatchSum(float scale, const ResizeAreaSt &st, const std::vector<const T *> &y_ptrs,
                       const std::vector<float> &y_scales, const ResizeAreaCachedInterpolation &x_interp,
                       float *&output_patch_ptr);
  uint32_t GetInputAndCheck(CpuKernelContext &ctx);
  DataType dtype_ = DT_INT8;
  std::vector<int64_t> in_shape1;
  std::vector<int64_t> in_shape2;
  std::vector<int64_t> out_shape;
  bool align_corners = false;
};
}  // namespace aicpu
#endif