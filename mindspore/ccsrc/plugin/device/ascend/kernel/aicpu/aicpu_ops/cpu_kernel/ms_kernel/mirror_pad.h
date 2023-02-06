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
#ifndef AICPU_KERNELS_NORMALIZED_MIRROR_PAD_H_
#define AICPU_KERNELS_NORMALIZED_MIRROR_PAD_H_
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "cpu_ops_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_log.h"
#include "securec.h"
#include "status.h"
#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {
class MirrorPadCpuKernel : public CpuKernel {
 public:
  MirrorPadCpuKernel() = default;
  ~MirrorPadCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  /**
   * @brief Init params
   * @param ctx cpu kernel context
   * @return status if success
   */
  template <typename T>
  uint32_t CheckAndInitParams(CpuKernelContext &ctx);

  /**
   * @brief padding
   * @param input_data_ptr ptr which store input data
   * @param output_data_ptr ptr which store output data
   * @return status if success
   */
  template <typename T>
  uint32_t MirrorPadCompute(T *input_data_ptr, T *output_data_ptr);

  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

 private:
  DataType data_type_;
};
}  // namespace aicpu
#endif
