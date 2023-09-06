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

#ifndef AICPU_KERNELS_NORMALIZED_CONCAT_H_
#define AICPU_KERNELS_NORMALIZED_CONCAT_H_

#include "cpu_ops_kernel.h"

#include "cpu_kernel/inc/cpu_ops_kernel.h"
#include "cpu_kernel/common/cpu_kernel_utils.h"
#include "common/kernel_log.h"
#include "common/status.h"
#include "securec/include/securec.h"

#include "unsupported/Eigen/CXX11/Tensor"

namespace aicpu {
const uint32_t NumIndices = 2;
template <typename T>
struct TTypes {
  // Rank-2 tensor (matrix) of scalar type T.
  using Matrix = Eigen::TensorMap<Eigen::Tensor<T, NumIndices, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>;

  using ConstMatrix =
    Eigen::TensorMap<Eigen::Tensor<const T, NumIndices, Eigen::RowMajor, Eigen::DenseIndex>, Eigen::Aligned>;
};

class ConcatCpuKernel : public CpuKernel {
 public:
  ConcatCpuKernel()
      : data_type_(DT_DOUBLE), input_dims_(0), n_(0), output_concat_dim_(0), axis_(0), inputs_flat_dim0_(0) {}

  ~ConcatCpuKernel() = default;

  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t CheckAndInitParams(const CpuKernelContext &ctx);

  template <typename T>
  uint32_t PrepareInput(const CpuKernelContext &ctx,
                        std::vector<std::shared_ptr<typename TTypes<T>::ConstMatrix>> &inputs);

  template <typename T>
  uint32_t PrepareOutput(const CpuKernelContext &ctx, std::shared_ptr<typename TTypes<T>::Matrix> &output);

  template <typename T>
  uint32_t DoCompute(const CpuKernelContext &ctx);

  template <typename T>
  uint32_t ConcatCompute(const CpuKernelContext &ctx,
                         const std::vector<std::shared_ptr<typename TTypes<T>::ConstMatrix>> &inputs,
                         std::shared_ptr<typename TTypes<T>::Matrix> &output);

 private:
  DataType data_type_;
  int32_t input_dims_;
  int64_t n_;
  int64_t output_concat_dim_;
  int64_t axis_;
  int64_t inputs_flat_dim0_;
};
}  // namespace aicpu
#endif
