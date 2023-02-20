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

#ifndef AICPU_KERNELS_NORMALIZED_GLU_H_
#define AICPU_KERNELS_NORMALIZED_GLU_H_

#include <memory>
#include <vector>

#include "unsupported/Eigen/CXX11/Tensor"
#include "securec.h"
#include "Eigen/Core"
#include "cpu_ops_kernel.h"
#include "cpu_kernel_utils.h"

namespace aicpu {

class GluCpuKernel : public CpuKernel {
 public:
  GluCpuKernel() : data_type_(DT_DOUBLE), split_dim_(-1), value_num_(0), value_dim_(0), value_data_ptr_(nullptr) {
    value_shape_vec_.clear();
  }

  ~GluCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  /**
   * @brief Init params
   * @param ctx cpu kernel context
   * @return status if success
   */
  uint32_t CheckAndInitParams(CpuKernelContext &ctx);

  /**
   * @brief split data when split dim is 0
   * @param input_data_ptr ptr which store input data
   * @param output_data_ptr ptr which store output data
   * @return status if success
   */
  template <typename T>
  uint32_t SplitWithDimZero(const CpuKernelContext &ctx, T *input_data_ptr, T *output_data_ptr);

  /**
   * @brief split data
   * @param input_data_ptr ptr which store input data
   * @param output_data_vec ptr which store output data
   * @return status if success
   */
  template <typename T>
  uint32_t SplitCompute(const CpuKernelContext &ctx, T *input_data_ptr, T *output_data_ptr);

  template <typename T>
  uint32_t DoCompute(CpuKernelContext &ctx);

 private:
  DataType x_;
  DataType data_type_;
  int32_t split_dim_;
  int64_t value_num_;
  int64_t value_dim_;
  void *value_data_ptr_;
  std::vector<int64_t> value_shape_vec_;
};
}  // namespace aicpu
#endif
