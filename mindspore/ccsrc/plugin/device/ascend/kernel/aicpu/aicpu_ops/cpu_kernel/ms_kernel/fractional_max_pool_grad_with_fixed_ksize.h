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
#ifndef AICPU_KERNELS_NORMALIZED_FRACTIONAL_MAX_POOL_GRAD_WITH_FIXED_KSIZE_H_
#define AICPU_KERNELS_NORMALIZED_FRACTIONAL_MAX_POOL_GRAD_WITH_FIXED_KSIZE_H_

#include <vector>
#include "cpu_ops_kernel.h"
#include "cpu_types.h"

namespace aicpu {
class FractionalMaxPoolGradWithFixedKsize : public CpuKernel {
 public:
  FractionalMaxPoolGradWithFixedKsize() = default;
  ~FractionalMaxPoolGradWithFixedKsize() override = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T>
  uint32_t FractionalMaxPoolGradWithFixedKsizeCompute(Tensor *out_backprop, Tensor *argmax, const int64_t data_nums,
                                                      const int n_size, const int c_size, const int input_h,
                                                      const int input_w, const int output_h, const int output_w,
                                                      CpuKernelContext &ctx);
  template <typename T>
  uint32_t ComputeSingleBatch(T *out_backprop_single_batch_addr, int64_t *argmax_single_batch_addr,
                              T *y_single_batch_addr, const int c_size, const int input_h, const int input_w,
                              const int output_h, const int output_w);
};
}  // namespace aicpu
#endif
