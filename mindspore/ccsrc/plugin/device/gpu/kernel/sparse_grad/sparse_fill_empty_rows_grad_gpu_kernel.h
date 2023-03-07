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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_FILL_EMPTY_ROWS_GRAD_H
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_FILL_EMPTY_ROWS_GRAD_H

#include <cuda_runtime_api.h>
#include <map>
#include <memory>
#include <vector>
#include <utility>
#include <algorithm>
#include <complex>

#include "abstract/utils.h"
#include "mindspore/core/ops/grad/sparse_fill_empty_rows_grad.h"
#include "plugin/device/gpu/hal/device/cuda_driver.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/sparse_fill_empty_rows_grad_impl.cuh"
constexpr size_t kInputsNum = 2;
constexpr size_t kOutputsNum = 2;

namespace mindspore {
namespace kernel {
class SparseFillEmptyRowsGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  SparseFillEmptyRowsGradGpuKernelMod() { ResetResource(); }
  ~SparseFillEmptyRowsGradGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    return kernel_func_(this, inputs, workspace, outputs, cuda_stream);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  void ResetResource() noexcept {
    output_dvalues_size_ = 0;
    output_ddefault_value_size_ = 0;
    workspace_flag_size_ = 0;
    workspace_sum_val_size_ = 0;
    is_null_input_ = false;
    input_size_list_.clear();
    workspace_size_list_.clear();
    output_size_list_.clear();
  }

 protected:
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &others) override;
  void InitSizeLists();

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *cuda_stream);
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  using SparseFillEmptyRowsGradFunc =
    std::function<bool(SparseFillEmptyRowsGradGpuKernelMod *, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, void *cuda_stream)>;
  static std::vector<std::pair<KernelAttr, SparseFillEmptyRowsGradFunc>> func_list_;
  SparseFillEmptyRowsGradFunc kernel_func_;
  bool is_null_input_{false};
  // input
  size_t reverse_map_size_{1};
  size_t grad_values_size_{1};
  size_t reverse_map_num_;
  size_t workspace_flag_size_;
  size_t workspace_sum_val_size_;
  size_t grad_values_num_;
  std::vector<int64_t> reverse_map_shape_;
  std::vector<int64_t> grad_values_shapes_;
  // output
  size_t output_dvalues_size_{1};
  size_t output_ddefault_value_size_{1};
  size_t dvalues_num_;
  std::vector<int64_t> dvalues_shapes_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_SPARSE_FILL_EMPTY_ROWS_GRAD_H
