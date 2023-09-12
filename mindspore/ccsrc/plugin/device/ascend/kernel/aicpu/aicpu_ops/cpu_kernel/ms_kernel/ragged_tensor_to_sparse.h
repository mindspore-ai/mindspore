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
#ifndef AICPU_KERNELS_NORMALIZED_RAGGED_TENSOR_TO_SPARSE_H_
#define AICPU_KERNELS_NORMALIZED_RAGGED_TENSOR_TO_SPARSE_H_

#include <securec.h>
#include <memory>
#include <vector>

#include "inc/cpu_ops_kernel.h"
#include "common/cpu_kernel_utils.h"
#include "common/kernel_log.h"
#include "common/status.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "utils/eigen_tensor.h"
#include "utils/kernel_util.h"

namespace aicpu {
class OpInputList {
 public:
  OpInputList() : ctx_(nullptr), start_(0), stop_(0) {}
  OpInputList(CpuKernelContext *ctx, uint32_t start, uint32_t stop) : ctx_(ctx), start_(start), stop_(stop) {}
  OpInputList(const OpInputList &) = default;
  OpInputList &operator=(const OpInputList &other) = default;
  Tensor *operator[](uint32_t i) const { return ctx_->Input(start_ + i); }
  uint32_t size() const { return stop_ - start_; }

 private:
  CpuKernelContext *ctx_;  // not owned
  uint32_t start_;
  uint32_t stop_;
};

class RaggedTensorToSparseCpuKernel : public CpuKernel {
 public:
  RaggedTensorToSparseCpuKernel() : type1(DT_DOUBLE), n_(1) {}
  ~RaggedTensorToSparseCpuKernel() = default;

 protected:
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  uint32_t CheckAndInitParams(const CpuKernelContext &ctx);

  template <typename T1>
  uint32_t ValidateInputs(std::vector<typename TTypes<T1>::Flat> rt_nested_splits, const Tensor *rt_dense_values_in);

  std::vector<std::vector<int64_t>> MakeIndexSuffixes(const TensorShape &values_shape);

  template <typename T1>
  bool IsCompleted(const std::vector<int64_t> &pos, int dim,
                   const std::vector<typename TTypes<T1>::Flat> &rt_nested_splits);

  void input_list(CpuKernelContext *ctx, OpInputList *list);

  template <typename T1, typename T2>
  uint32_t DoCompute(CpuKernelContext *ctx);

  template <typename T1>
  uint32_t Update(const CpuKernelContext &ctx, std::vector<typename TTypes<T1>::Flat> rt_nested_splits);

  template <typename T2>
  void OutPutSparseValues(const CpuKernelContext &ctx);

  template <typename T1>
  void OutPutSparseDenseShape(const CpuKernelContext &ctx, OpInputList rt_nested_splits_in,
                              std::vector<typename TTypes<T1>::Flat> rt_nested_splits);

  uint32_t ComputeWithSplitTypeInt32(CpuKernelContext *ctx);

  uint32_t ComputeWithSplitTypeInt64(CpuKernelContext *ctx);

 private:
  DataType type1;
  int64_t n_;
};
}  // namespace aicpu
#endif
