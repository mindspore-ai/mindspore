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
#ifndef AICPU_KERNELS_NORMALIZED_CROPANDRESIZEGRADIMAGE_H_
#define AICPU_KERNELS_NORMALIZED_CROPANDRESIZEGRADIMAGE_H_

#include "Eigen/Core"
#include "cpu_ops_kernel.h"
#include "utils/bcast.h"

namespace aicpu {
class CropAndResizeGradImageCpuKernel : public CpuKernel {
 public:
  CropAndResizeGradImageCpuKernel() = default;
  ~CropAndResizeGradImageCpuKernel() = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t cheakInputTypeAndGetDatas(CpuKernelContext &ctx);

  template <typename T>
  uint32_t GradOfImageCompute(CpuKernelContext &ctx, int64_t start, int64_t end);
  template <typename T>
  uint32_t GradOfImageComputeShared(CpuKernelContext &ctx);

  std::vector<int64_t> grads_shape_;
  std::vector<int64_t> image_size_shape_;
  std::vector<int64_t> boxes_shape_;
  std::vector<int64_t> box_ind_shape_;
  std::vector<int64_t> output_shape_;
  DataType data_type_;
};
}  // namespace aicpu
#endif