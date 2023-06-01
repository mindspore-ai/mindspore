/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021-2022. All rights reserved.
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

#ifndef AICPU_KERNELS_NORMALIZED_UPSAMPLE_NEAREST3D_GRAD_H
#define AICPU_KERNELS_NORMALIZED_UPSAMPLE_NEAREST3D_GRAD_H

#include "cpu_ops_kernel.h"
namespace aicpu {
class UpsampleNearest3dGradCpuKernel : public CpuKernel {
 public:
  ~UpsampleNearest3dGradCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t UpsampleNearest3dGradParamCheck(CpuKernelContext &ctx);

  template <typename T, typename S>
  uint32_t UpsampleNearest3dGradCompute(const CpuKernelContext &ctx);

  template <typename T, typename S>
  void InnerCompute(int64_t c, const T *grad_output_data, S *grad_input_ptr);

  int64_t input_depth;
  int64_t input_height;
  int64_t input_width;
  int64_t output_depth;
  int64_t output_height;
  int64_t output_width;
  int64_t output_slice_size;
  int64_t input_slice_size;

  std::vector<float> scales;
  std::vector<int64_t> none_list;
  std::vector<int64_t> grads_input_shape;
  std::vector<int64_t> grads_output_shape;
};
}  // namespace aicpu
#endif
