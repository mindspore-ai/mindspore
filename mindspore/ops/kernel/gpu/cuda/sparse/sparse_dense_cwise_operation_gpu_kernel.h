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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_DENSE_CWISE_OPERATION_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_DENSE_CWISE_OPERATION_GPU_KERNEL_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/sparse_dense_cwise_operation_impl.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
constexpr size_t MAX_DIMS = 5;
class SparseDenseCwiseOperationGpuKernelMod : public NativeGpuKernelMod,
                                              public MatchKernelHelper<SparseDenseCwiseOperationGpuKernelMod> {
 public:
  SparseDenseCwiseOperationGpuKernelMod() {
    dimension_ = 0;
    value_num_ = 0;
    dense_num_ = 1;
    is_null_input_ = false;
    data_unit_size_ = 0;
    stream_ptr_ = nullptr;
  }
  ~SparseDenseCwiseOperationGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    stream_ptr_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  bool GetOpType();
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  using SupportList = std::vector<std::pair<KernelAttr, SparseDenseCwiseOperationGpuKernelMod::KernelRunFunc>>;
  void ResetResource();
  void InitSizeLists();
  void *stream_ptr_;

  bool is_null_input_;
  int64_t dimension_;
  int64_t value_num_;
  int64_t dense_num_;
  SparseDenseCwiseOperationFunctionType op_func_type_{SPARSE_DENSE_CWISE_OPERATION_INVALID_TYPE};
  std::vector<int64_t> dense_shape_;
  std::vector<int64_t> i_ = {1, 1, 1, 1, 1, 1, 1};
  std::vector<int64_t> o_ = {1, 1, 1, 1, 1, 1, 1};
  // default values
  size_t data_unit_size_; /* size of T */
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_SPARSE_SPARSE_DENSE_CWISE_OPERATION_GPU_KERNEL_H_
