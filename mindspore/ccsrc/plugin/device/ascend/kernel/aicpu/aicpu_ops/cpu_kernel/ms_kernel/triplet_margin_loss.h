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

#ifndef AICPU_KERNELS_NORMALIZED_TRIPLET_MARGIN_LOSS_H_
#define AICPU_KERNELS_NORMALIZED_TRIPLET_MARGIN_LOSS_H_

#include "cpu_ops_kernel.h"

namespace aicpu {
class TripletMarginLossCpuKernel : public CpuKernel {
 public:
  TripletMarginLossCpuKernel() = default;
  ~TripletMarginLossCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  static uint32_t TripletMarginLossComputeRealType(
    CpuKernelContext &ctx, int p_value, float margin_value, float eps_value, bool swap_value,
    std::string reduction_value, int64_t num_elements, int64_t data_num_output_reduction_none,
    int64_t data_num_each_batch_input, int64_t data_num_each_batch_output_reduction_none, int64_t batch_size,
    int64_t once_compute_size, bool broadcast, std::vector<int64_t> x_reshape_vector,
    std::vector<int64_t> positive_reshape_vector, std::vector<int64_t> negative_reshape_vector);

  template <typename T>
  static uint32_t TripletMarginLossComputeRealTypeFloat16(
    CpuKernelContext &ctx, int p_value, float margin_value, float eps_value, bool swap_value,
    std::string reduction_value, int64_t num_elements, int64_t data_num_output_reduction_none,
    int64_t data_num_each_batch_input, int64_t data_num_each_batch_output_reduction_none, int64_t batch_size,
    int64_t once_compute_size, bool broadcast, std::vector<int64_t> x_reshape_vector,
    std::vector<int64_t> positive_reshape_vector, std::vector<int64_t> negative_reshape_vector);

  template <typename T>
  static uint32_t TripletMarginLossComputeComplexType(
    CpuKernelContext &ctx, int p_value, float margin_value, float eps_value, bool swap_value,
    std::string reduction_value, int64_t num_elements, int64_t data_num_output_reduction_none,
    int64_t data_num_each_batch_input, int64_t data_num_each_batch_output_reduction_none, int64_t batch_size,
    int64_t once_compute_size, bool broadcast, std::vector<int64_t> x_reshape_vector,
    std::vector<int64_t> positive_reshape_vector, std::vector<int64_t> negative_reshape_vector);
};
}  // namespace aicpu
#endif  // namespace aicpu
