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

#ifndef AICPU_KERNELS_NORMALIZED_UPSAMPLE_TRILINEAR3D_GRAD_H
#define AICPU_KERNELS_NORMALIZED_UPSAMPLE_TRILINEAR3D_GRAD_H

#include <string>
#include <vector>

#include "cpu_ops_kernel.h"
namespace aicpu {
class UpsampleTrilinear3dGradCpuKernel : public CpuKernel {
 public:
  ~UpsampleTrilinear3dGradCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  struct WeightsAndIndices {
    void operator()(int64_t *const input_index0, int64_t *const input_index1, T *const lambda_0, T *const lambda_1) {
      *input_index0 = id0;
      *input_index1 = id1;
      *lambda_0 = lambda0;
      *lambda_1 = lambda1;
    }
    void Step(const int64_t stride) {
      id0 *= stride;
      id1 *= stride;
    }
    int64_t id0;
    int64_t id1;
    T lambda0;
    T lambda1;
  };
  template <typename S>
  void ComputeWeightsAndIndices(WeightsAndIndices<S> &wi, S scale, int64_t out_idx, int64_t input_size,
                                int64_t output_size, int64_t stride);

  template <typename S>
  void ComputeHelper(const CpuKernelContext &ctx, std::vector<WeightsAndIndices<S>> &helper, S scale,
                     int64_t input_size, int64_t output_size, int64_t stride);

  uint32_t UpsampleTrilinear3dGradParamCheck(CpuKernelContext &ctx);

  template <typename T, typename S>
  uint32_t UpsampleTrilinear3dGradCompute(const CpuKernelContext &ctx);

  template <typename T, typename S, typename R>
  uint32_t RealCompute(const CpuKernelContext &ctx, T *const grad_output_ptr, R *const grad_input_ptr);

  int64_t FetchBlockSize(const CpuKernelContext &ctx, const int64_t parallel_num, const int64_t cost);

  int64_t channels;
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
  bool align_corners = false;
};
}  // namespace aicpu
#endif
