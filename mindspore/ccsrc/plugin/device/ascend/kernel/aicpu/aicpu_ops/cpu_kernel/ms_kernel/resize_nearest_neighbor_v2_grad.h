/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
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

#ifndef AICPU_KERNELS_NORMALIZED_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_H
#define AICPU_KERNELS_NORMALIZED_RESIZE_NEAREST_NEIGHBOR_V2_GRAD_H

#include <unordered_map>
#include <vector>

#include "inc/cpu_ops_kernel.h"
#include "utils/eigen_tensor.h"

namespace aicpu {
constexpr uint32_t kValue4 = 4;
class ResizeNearestNeighborV2GradCpuKernel : public CpuKernel {
 public:
  ~ResizeNearestNeighborV2GradCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t ResizeNearestNeighborV2GradParamCheck(CpuKernelContext &ctx);

  template <typename T>
  uint32_t ResizeNearestNeighborV2GradCompute(const CpuKernelContext &ctx);

  template <typename T, typename S>
  void RealCompute(T *const grads_4d, S *const y_4d);

  size_t y_size{0};
  bool align_corners;
  bool half_pixel_centers;
  std::vector<int64_t> grads_shape{};
  std::vector<int64_t> y_shape{};
  Eigen::Index batch_size;
  Eigen::Index in_height;
  Eigen::Index in_width;
  Eigen::Index channels;

  Eigen::Index out_height;
  Eigen::Index out_width;
};
}  // namespace aicpu
#endif
