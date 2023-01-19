/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_SPLIT_H_
#define AICPU_KERNELS_NORMALIZED_SPLIT_H_

#include <memory>
#include <vector>

#include "unsupported/Eigen/CXX11/Tensor"
#include "securec.h"

#include "cpu_ops_kernel.h"
#include "cpu_kernel_utils.h"
#include "kernel_log.h"
#include "status.h"

namespace aicpu {
class SplitCpuKernel : public CpuKernel {
 public:
  SplitCpuKernel() = default;
  ~SplitCpuKernel() override = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  /**
   * @brief Init params
   * @param ctx cpu kernel context
   * @return status if success
   */
  uint32_t CheckAndInitParams(CpuKernelContext &ctx);
  /**
   * @brief split data when split num is 1
   * @param input_data_ptr ptr which store input data
   * @param output_data_vec vector which store all output data ptr
   * @return status if success
   */
  template <typename T>
  uint32_t SplitWithOneOutput(T *input_data_ptr, std::vector<T *> output_data_vec);
  /**
   * @brief split data when split dim is 0
   * @param input_data_ptr ptr which store input data
   * @param output_data_vec vector which store all output data ptr
   * @return status if success
   */
  template <typename T>
  uint32_t SplitWithDimZero(T *input_data_ptr, std::vector<T *> output_data_vec);
  /**
   * @brief split data
   * @param input_data_ptr ptr which store input data
   * @param output_data_vec vector which store all output data ptr
   * @return status if success
   */
  template <typename T>
  uint32_t SplitCompute(T *input_data_ptr, std::vector<T *> output_data_vec);

  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

 private:
  DataType data_type_;
  int32_t split_dim_;
  int64_t num_split_;
  int64_t value_num_;
  void *value_data_ptr_;
  std::vector<void *> output_ptr_vec_;
  std::vector<int64_t> value_shape_vec_;
};
}  // namespace aicpu
#endif