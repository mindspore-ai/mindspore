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

#ifndef AICPU_KERNELS_NORMALIZED_RESIZE_NEAREST_NEIGHBOR_V2_H
#define AICPU_KERNELS_NORMALIZED_RESIZE_NEAREST_NEIGHBOR_V2_H

#include <unordered_map>

#include "cpu_ops_kernel.h"
#include "utils/eigen_tensor.h"

namespace aicpu {
constexpr uint32_t kValue4 = 4;
class ResizeNearestNeighborV2CpuKernel : public CpuKernel {
 public:
  ~ResizeNearestNeighborV2CpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t ResizeNearestNeighborV2ParamCheck(CpuKernelContext &ctx);
  template <typename T>
  uint32_t ResizeNearestNeighborV2Compute(CpuKernelContext &ctx);
  template <typename T>
  void InnerCompute(
    Eigen::Index b, Eigen::Index y,
    Eigen::TensorMap<Eigen::Tensor<T, kValue4, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> x_4d,
    Eigen::TensorMap<Eigen::Tensor<T, kValue4, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned> y_4d);

  std::string data_format = "NHWC";
  std::unordered_map<char, size_t> dim_idx_map_;
  bool align_corners;
  bool half_pixel_centers;
  Eigen::Index batch_size;
  Eigen::Index in_height;
  Eigen::Index in_width;
  Eigen::Index channels;
  Eigen::Index out_height;
  Eigen::Index out_width;
  float height_scale;
  float width_scale;
};
}  // namespace aicpu
#endif
