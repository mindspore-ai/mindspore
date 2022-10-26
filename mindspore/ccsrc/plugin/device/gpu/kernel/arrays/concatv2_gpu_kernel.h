/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_CONCATV2_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_CONCATV2_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <map>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/concatv2_impl.cuh"

namespace mindspore {
namespace kernel {
class ConcatV2FwdGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<ConcatV2FwdGpuKernelMod> {
 public:
  ConcatV2FwdGpuKernelMod() = default;
  ~ConcatV2FwdGpuKernelMod() override = default;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

  int axis_{0};
  int ori_axis_{0};
  int input_num_{1};
  size_t output_size_{0};
  int all_size_before_axis_{1};
  int all_size_axis_{1};
  std::string kernel_name_{"ConcatV2"};
  std::vector<int> not_null_input_index_;
  std::vector<int> len_axis_;
  std::vector<void *> inputs_host_;
  std::string origin_data_format_;
  void *stream_ptr_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_ARRAYS_CONCATV2_GPU_KERNEL_H_
