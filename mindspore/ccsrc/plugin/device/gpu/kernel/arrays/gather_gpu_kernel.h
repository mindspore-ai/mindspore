/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_GATHER_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_GATHER_GPU_KERNEL_H_

#include <vector>
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/gather.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

namespace mindspore {
namespace kernel {
class GatherFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  GatherFwdGpuKernelMod() {}
  ~GatherFwdGpuKernelMod() = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (!kernel_func_) {
      MS_LOG(ERROR) << "GatherFwdGpu's kernel function is not initialized.";
      return false;
    }
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    if (is_null_input_) {
      return true;
    }
    VARIABLE_NOT_USED(workspace);

    auto input_addr = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
    auto index_addr = reinterpret_cast<S *>(inputs.at(index_idx_)->addr);
    auto output_addr = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);

    const size_t k2Idx = 2;
    const size_t k3Idx = 3;
    Gather(input_addr, index_addr, output_addr, dims_[0], dims_[1], dims_[k2Idx], dims_[k3Idx], cuda_stream,
           GET_CTX_DEVICE_ID);
    return true;
  }

  bool SetDimParam(int64_t dim_value);

  using GatherFwdFunc =
    std::function<bool(GatherFwdGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, void *)>;

  GatherFwdFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, GatherFwdFunc>> func_list_;

  std::vector<size_t> input_shapes_;
  std::vector<size_t> index_shapes_;
  std::vector<size_t> output_shapes_;

  size_t dims_[4] = {};
  bool is_null_input_{false};
  bool is_dynamic_case_{false};
  size_t index_idx_{1};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_GATHER_GPU_KERNEL_H_
